from typing import Tuple, List, Union, Callable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

import dualpipe.comm as comm
from dualpipe.utils import WeightGradStore, run_backward, scatter, gather

class DualPipe(nn.Module):
    '''
    DualPipe optimizes performance through several innovative techniques:
        Bidirectional Pipelining:
            - Unlike traditional unidirectional pipelines, DualPipe processes data in both forward and reverse directions simultaneously across the pipeline ranks. For example, the first half of ranks process forward direction in phase 0 and reverse in phase 1, while the second half do the opposite. This keeps more GPUs active at once, reducing idle time.
        Overlapped Computations:
            - When modules support overlapped_forward_backward, the _forward_backward_compute_chunk method computes forward and backward passes concurrently. This leverages GPU parallelism, reducing the total time by overlapping operations that would otherwise be sequential.
        Efficient Scheduling:
            - The step method uses an 8-step schedule (nF0, nF0F1, nB1W1F1, nF0B1F1B0, nB1F1B0, nB1B0, nWB0, nW) that interleaves forward, backward, and weight update tasks. This minimizes dependencies and maximizes resource utilization, as seen in the main step (nF0B1F1B0), where all four operations overlap.
        Asynchronous Communication:
            - Communication operations (_recv_forward, _send_forward, _recv_backward, _send_backward) use PyTorch’s asynchronous irecv and isend. The _commit_and_wait_comm method batches these operations, allowing computation to proceed while communication occurs in the background, reducing blocking time.
        Zero-Bubble Optimization:
            - In steps like nB1W1F1, nB1B0, and nWB0, the enable_zb flag activates zero-bubble techniques via WeightGradStore. This delays gradient application until optimal points in the schedule, aligning computation and communication to eliminate idle periods.
    These combined strategies result in a highly efficient pipeline that maximizes GPU utilization and minimizes latency.
    '''
    def __init__(
        self,
        modules: Tuple[nn.Module, nn.Module],
        batch_dim: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
        rank_mapping: Optional[List[int]] = None,
    ) -> None:
        '''
        Purpose: Sets up the pipeline with two modules and configures distributed communication.
        
        Parameters:
            - modules: A tuple of two nn.Module instances, one for each direction.
            - batch_dim: Dimension along which to split batches (default: 0).
            - process_group: PyTorch distributed process group (defaults to the global group).
            - rank_mapping: Maps process group ranks to pipeline ranks (defaults to sequential mapping).
        
        Key Logic:
            - Verifies that modules are on the current GPU.
            - Checks if modules support overlapped forward-backward computation via overlapped_forward_backward.
            - Determines the rank’s position (first, last, middle, or second half) in the pipeline for directional logic.
        '''
        super().__init__()

        ## Verifies that modules are on the current GPU.
        assert next(modules[0].parameters()).device == torch.device(torch.cuda.current_device())

        ## Checks if modules support overlapped forward-backward computation via overlapped_forward_backward.
        self.module = nn.ModuleList(modules)
        self.overlapped_forward_backward = type(modules[0]) == type(modules[1]) and hasattr(type(modules[0]), "overlapped_forward_backward")
        self.batch_dim = batch_dim
        self.group = process_group or dist.distributed_c10d._get_default_group()
        self.num_ranks = self.group.size()

        ## Determines the rank’s position (first, last, middle, or second half) in the pipeline for directional logic.
        # rank_mapping: Map rank in process_group to actual pp rank.
        # rank_inverse_mapping: Map actual pp rank to rank in process_group.
        if rank_mapping is None:
            rank_mapping = list(range(self.num_ranks))
        rank_inverse_mapping = [None] * (self.num_ranks + 1)
        for i in range(self.num_ranks):
            rank_inverse_mapping[rank_mapping[i]] = i

        self.rank = rank_mapping[self.group.rank()]
        self.first_rank = rank_inverse_mapping[0]
        self.prev_rank = rank_inverse_mapping[self.rank - 1]
        self.next_rank = rank_inverse_mapping[self.rank + 1]
        self.last_rank = rank_inverse_mapping[self.num_ranks - 1]

        self.is_first_rank = self.rank == 0
        self.is_last_rank = self.rank == self.num_ranks - 1
        self.is_in_second_half = self.rank >= self.num_ranks // 2
        self.is_middle_rank = (self.rank == self.num_ranks // 2 - 1) or (self.rank == self.num_ranks // 2)

    def _reset_states(self) -> None:
        '''
        Purpose: Initializes or resets internal buffers for each step.

        Key Variables:
            - input_chunks, output_chunks, input_grad_chunks, output_grad_chunks: Buffers for inputs, outputs, and gradients in both phases (0 and 1).
            - labels, loss_chunks, criterion: For loss computation at endpoint ranks.
            - current_f_chunk_id, current_b_chunk_id: Track micro-batch indices for forward and backward passes.
            - comm_ops, to_free: Manage communication operations and tensor cleanup.
        '''
        WeightGradStore.clear()

        self.input_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.output_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.input_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.output_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.labels: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = None
        self.loss_chunks: List[torch.Tensor] = []
        self.criterion: Callable = None

        self.current_f_chunk_id: List[int] = [0, 0]
        self.current_b_chunk_id: List[int] = [0, 0]
        self.current_send_f_chunk_id: List[int] = [0, 0]
        self.current_send_b_chunk_id: List[int] = [0, 0]
        self.current_recv_f_chunk_id: List[int] = [0, 0]
        self.current_recv_b_chunk_id: List[int] = [0, 0]
        self.comm_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []

    def _forward_compute_chunk(self, phase: int) -> None:
        '''
        Forward Computation
        
        Purpose: Executes a forward pass for a micro-batch in a given phase (0 or 1).
        
        Logic:
            - Flips phase based on rank’s position (first or second half).
            - Processes inputs through the module, computes loss if at the last stage and a criterion is provided.
            - Stores outputs unless it’s the last stage and outputs aren’t needed.
        '''
        ## Flips phase based on rank’s position (first or second half).
        phase ^= self.is_in_second_half
        chunk_id = self.current_f_chunk_id[phase]
        self.current_f_chunk_id[phase] += 1
        inputs = self.input_chunks[phase][chunk_id]
        if self.forward_only:
            self.input_chunks[phase][chunk_id] = None

        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)

        ## Processes inputs through the module, computes loss if at the last stage and a criterion is provided.
        outputs = self.module[phase](*inputs)

        ## Stores outputs unless it’s the last stage and outputs aren’t needed.
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
        if is_last_stage and self.criterion is not None:
            labels = self.labels[phase][chunk_id]
            loss = self.criterion(*outputs, *labels)
            self.loss_chunks.append(loss)

        if (not is_last_stage) or self.return_outputs:
            self.output_chunks[phase].append(outputs)

    def _backward_compute_chunk(self, phase: int, enable_zb: bool = False) -> None:
        '''
        Backward Computation
        
        Purpose: Executes a backward pass for a micro-batch.
        
        Logic:
            - Skips if in forward-only mode.
            - Uses WeightGradStore for zero-bubble optimization when enabled.
            - At the last stage, computes gradients from loss; otherwise, backpropagates gradients from output_grads.
            - Stores input gradients for communication.
        '''
        ## Skips if in forward-only mode.
        if self.forward_only:
            return

        phase ^= self.is_in_second_half
        chunk_id = self.current_b_chunk_id[phase]
        self.current_b_chunk_id[phase] += 1

        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)

        ## Uses WeightGradStore for zero-bubble optimization when enabled.
        WeightGradStore.enabled = enable_zb

        if is_last_stage:
            ## At the last stage, computes gradients from loss;
            loss = self.loss_chunks[chunk_id]
            loss.backward()
            loss.detach_()
        else:
            ## otherwise, backpropagates gradients from output_grads.
            outputs = self.output_chunks[phase][chunk_id]
            if not self.return_outputs:
                self.output_chunks[phase][chunk_id] = None
            output_grads = self.output_grad_chunks[phase][chunk_id]
            self.output_grad_chunks[phase][chunk_id] = None
            non_empty = [(t, g) for t, g in zip(outputs, output_grads) if g is not None]
            outputs, output_grads = list(zip(*non_empty))
            if len(outputs) > 0:
                run_backward(outputs, output_grads)
        WeightGradStore.enabled = False
        if enable_zb:
            WeightGradStore.flush()

        ## Stores input gradients for communication.
        inputs = self.input_chunks[phase][chunk_id]
        self.input_chunks[phase][chunk_id] = None
        input_grads = [t.grad for t in inputs]
        self.input_grad_chunks[phase].append(input_grads)

    def _forward_backward_compute_chunk(self, phase0: int, phase1: int) -> None:
        '''
        Overlapped Forward-Backward
        Purpose: Overlaps forward and backward computations when supported by the modules.
        
        Logic:
            - Prepares inputs and outputs for both phases.
            - Calls the module’s overlapped_forward_backward method to compute both simultaneously.
            - Handles post-processing (storing outputs, gradients, and losses).
        '''
        ## Prepares inputs and outputs for both phases.
        if self.forward_only:
            self._forward_compute_chunk(phase0)
            return

        if not self.overlapped_forward_backward:
            self._forward_compute_chunk(phase0)
            self._backward_compute_chunk(phase1)
            return

        # pre-forward
        phase0 ^= self.is_in_second_half
        chunk_id0 = self.current_f_chunk_id[phase0]
        self.current_f_chunk_id[phase0] += 1
        module0 = self.module[phase0]
        inputs0 = self.input_chunks[phase0][chunk_id0]
        is_last_stage0 = (self.is_first_rank and phase0 == 1) or (self.is_last_rank and phase0 == 0)

        if is_last_stage0 and self.criterion is not None:
            labels0 = self.labels[phase0][chunk_id0]
            criterion0 = self.criterion
        else:
            labels0 = []
            criterion0 = None

        # pre-backward
        phase1 ^= self.is_in_second_half
        chunk_id1 = self.current_b_chunk_id[phase1]
        self.current_b_chunk_id[phase1] += 1
        module1 = self.module[phase1]
        is_last_stage1 = (self.is_first_rank and phase1 == 1) or (self.is_last_rank and phase1 == 0)

        if is_last_stage1:
            loss1 = self.loss_chunks[chunk_id1]
            outputs1 = []
            output_grads1 = []
        else:
            loss1 = None
            outputs1 = self.output_chunks[phase1][chunk_id1]
            if not self.return_outputs:
                self.output_chunks[phase1][chunk_id1] = None
            output_grads1 = self.output_grad_chunks[phase1][chunk_id1]
            self.output_grad_chunks[phase1][chunk_id1] = None
            non_empty = [(t, g) for t, g in zip(outputs1, output_grads1) if g is not None]
            outputs1, output_grads1 = list(zip(*non_empty))

        ## Calls the module’s overlapped_forward_backward method to compute both simultaneously.
        # forward & backward
        outputs0, loss0 = type(module0).overlapped_forward_backward(
            module0, inputs0, criterion0, labels0,
            module1, loss1, outputs1, output_grads1,
        )

        ## Handles post-processing (storing outputs, gradients, and losses).
        # post-forward
        if (not is_last_stage0) or self.return_outputs:
            self.output_chunks[phase0].append(outputs0)
        if is_last_stage0 and self.criterion is not None:
            self.loss_chunks.append(loss0)

        # post-backward
        inputs = self.input_chunks[phase1][chunk_id1]
        self.input_chunks[phase1][chunk_id1] = None
        input_grads1 = [t.grad for t in inputs]
        self.input_grad_chunks[phase1].append(input_grads1)

    '''
    Pipeline Steps
    Purpose: Wrap computation and communication for specific pipeline stages.
    
    Logic:
        - _forward_chunk: Receives inputs, computes forward, sends outputs.
        - _backward_chunk: Receives gradients, computes backward, sends input gradients, optionally with zero-bubble optimization.
        - _forward_backward_chunk: Overlaps forward and backward for two phases with communication.
        - _weight_chunk: Applies gradient updates using WeightGradStore.
    '''
    def _forward_chunk(self, phase: int, recv: bool = True, send: bool = True) -> None:
        if recv:
            self._recv_forward(phase)
        self._commit_and_wait_comm()

        self._forward_compute_chunk(phase)

        if send:
            self._send_forward(phase)

    def _backward_chunk(self, phase: int, enable_zb: bool = False, recv: bool = True, send: bool = True) -> None:
        if recv:
            self._recv_backward(phase)
        self._commit_and_wait_comm()

        self._backward_compute_chunk(phase, enable_zb)

        if send:
            self._send_backward(phase)

    def _forward_backward_chunk(self, phase0: int, phase1: int, recv0: bool = True) -> None:
        if recv0:
            self._recv_forward(phase0)
        self._recv_backward(phase1)
        self._commit_and_wait_comm()

        self._forward_backward_compute_chunk(phase0, phase1)

        self._send_forward(phase0)
        self._send_backward(phase1)

    def _weight_chunk(self) -> None:
        if self.forward_only:
            return

        self._commit_and_wait_comm()

        # Assume FIFO
        WeightGradStore.pop()

    def _free_tensors(self) -> None:
        for tensor in self.to_free:
            assert tensor._base is None, f"pipeline stage should not return view tensors {dist.get_rank(), tensor.shape}"
            tensor.data = torch.Tensor()
        self.to_free = []
        
    '''
    Communication Helpers:
        - _recv_forward, _send_forward: Receive and send forward pass tensors between ranks.
        - _recv_backward, _send_backward: Handle backward pass gradient communication.
        - _commit_and_wait_comm: Executes queued communication operations (send/receive) and frees tensors.
    '''
    def _recv_forward(self, phase: int) -> None:
        phase ^= self.is_in_second_half
        is_first_stage = (self.is_first_rank and phase == 0) or (self.is_last_rank and phase == 1)
        if is_first_stage:
            return

        self.current_recv_f_chunk_id[phase] += 1
        tensors = comm.append_irecv(self.comm_ops, self.prev_rank if phase == 0 else self.next_rank, self.group)
        self.input_chunks[phase].append(tensors)

    def _send_forward(self, phase: int) -> None:
        phase ^= self.is_in_second_half
        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)
        if is_last_stage:
            return

        chunk_id = self.current_send_f_chunk_id[phase]
        self.current_send_f_chunk_id[phase] += 1
        tensors = self.output_chunks[phase][chunk_id]

        comm.append_isend(self.comm_ops, tensors, self.next_rank if phase == 0 else self.prev_rank, self.group)

        if not self.return_outputs:
            self.to_free.extend(tensors)

    def _recv_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        phase ^= self.is_in_second_half
        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)
        if is_last_stage:
            return

        self.current_recv_b_chunk_id[phase] += 1
        tensors = comm.append_irecv(self.comm_ops, self.next_rank if phase == 0 else self.prev_rank, self.group)
        self.output_grad_chunks[phase].append(tensors)

    def _send_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        phase ^= self.is_in_second_half
        is_first_stage = (self.is_first_rank and phase == 0) or (self.is_last_rank and phase == 1)
        if is_first_stage:
            return

        chunk_id = self.current_send_b_chunk_id[phase]
        self.current_send_b_chunk_id[phase] += 1
        tensors = self.input_grad_chunks[phase][chunk_id]
        self.input_grad_chunks[phase][chunk_id] = None

        comm.append_isend(self.comm_ops, tensors, self.prev_rank if phase == 0 else self.next_rank, self.group)

    def _commit_and_wait_comm(self) -> None:
        if not self.comm_ops:
            return
        reqs = dist.batch_isend_irecv(self.comm_ops)
        for req in reqs:
            req.wait()
        self.comm_ops = []
        self._free_tensors()

    def step(
        self,
        *inputs: Optional[torch.Tensor],
        num_chunks: int = 0,
        criterion: Optional[Callable] = None,
        labels: List[Optional[torch.Tensor]] = [],
        return_outputs: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
        """
        Execute a training or inference step.

        Arguments:
            *inputs: Module inputs. Required only on the first/last ranks.
            num_chunks: The number of micro-batches.
            criterion: Loss function, invoked as ``criterion(*outputs, *labels)``. Required only on the first/last ranks.
            labels: Labels of the loss function. Required only on the first/last ranks.
                labels on the first rank corresponds to inputs on the last rank.
                labels on the last rank corresponds to inputs on the first rank.
            return_outputs: Whether to return outputs on the first/last ranks. Default: ``False``.

        Returns: (loss, outputs)
            loss: Loss for the batch.
                loss on the first rank corresponds to inputs on the last rank.
                loss on the last rank corresponds to inputs on the first rank.
                Otherwise: ``None``.
            outputs: Returned only if ``return_outputs=True``.
                outputs on the first rank corresponds to inputs on the last rank.
                outputs on the last rank corresponds to inputs on the first rank.
                Otherwise: ``None``.

        """

        '''
        Main Execution (step)
        Purpose: Orchestrates the entire training or inference step across micro-batches.
        
        Parameters:
            - inputs, labels: Provided at first/last ranks.
            - num_chunks: Number of micro-batches (must be even and ≥ 2 × num_ranks).
            - criterion: Loss function for endpoint ranks.
            - return_outputs: Whether to return outputs from endpoint ranks.
        
        Logic:
            - Splits inputs/labels into micro-batches using scatter.
            - Assigns inputs/labels based on rank (first rank: forward inputs, reverse labels; last rank: vice versa).
            - Executes 8 steps based on the bidirectional schedule (detailed below).
            - Returns loss and outputs (if requested) after gathering results.

        Steps:
            1. nF0: Forward computations for phase 0 (warm-up).
            2. nF0F1: Alternates forward for phases 0 and 1 with communication.
            3. nB1W1F1: Backward for phase 1, weight updates, and forward for phase 1 with zero-bubble optimization.
            4. nF0B1F1B0: Main overlap phase for forward and backward in both directions.
            5. nB1F1B0: Backward for phase 1 with overlapped forward-backward for phase 0.
            6. nB1B0: Backward for both phases, with zero-bubble for the second half.
            7. nWB0: Weight updates and backward for phase 0 with zero-bubble.
            8. nW: Final weight updates.


            n:
                the number of times a particular operation or sequence of operations is repeated in the pipeline schedule. 
                For example, in nF0, "n" indicates that the forward operation in phase 0 (F0) is performed "n" times. 
                It’s a multiplier for the operations that follow it.
            F0: 
                the forward pass in phase 0. In bidirectional pipelining, phase 0 typically refers to 
                one direction of data flow—often the forward direction for the first half of the pipeline ranks 
                (e.g., processing units like GPUs).
            F1: 
                the forward pass in phase 1. Phase 1 represents the opposite direction of data flow compared to 
                phase 0—for example, the reverse direction for the first half of the ranks or the forward direction 
                for the second half.
            B0: 
                the backward pass in phase 0. It corresponds to the backward pass (gradient computation) for the data flowing 
                in the direction associated with phase 0.
            B1: 
                the backward pass in phase 1. It corresponds to the backward pass for the data flowing in the direction 
                associated with phase 1.
            W:
                the weight update operation. It involves applying the gradients computed during the backward pass (B0 or B1) 
                to update the model parameters. 
            W1: 
                a specific weight update related to phase 1. It likely occurs after the backward pass in phase 1 (B1), 
                as seen in nB1W1F1. It could represent a targeted or partial weight update tied to phase 1’s computations.

        '''

        assert comm.TENSOR_SHAPES is not None and comm.TENSOR_DTYPE is not None, \
            "You need to call set_p2p_tensor_shapes and set_p2p_tensor_dtype before doing a step."
        self.forward_only = not torch.is_grad_enabled()
        self.return_outputs = return_outputs

        rank = self.rank
        num_ranks = self.num_ranks
        assert num_ranks % 2 == 0
        assert num_chunks > 0 and num_chunks % 2 == 0 and num_chunks >= num_ranks * 2, f"{num_chunks=}, {num_ranks=}"
        num_half_ranks = num_ranks // 2
        half_rank = min(rank, num_ranks - 1 - rank)
        half_num_chunks = num_chunks // 2
        self.num_half_ranks = num_half_ranks
        self.half_rank = half_rank

        if not self.forward_only and (self.is_first_rank or self.is_last_rank):
            assert criterion is not None

        self._reset_states()

        inputs = scatter(inputs, half_num_chunks, self.batch_dim)
        labels = scatter(labels, half_num_chunks, self.batch_dim)
        if self.is_first_rank:
            self.input_chunks = (inputs, [])
            self.labels = ([], labels)
        elif self.is_last_rank:
            self.input_chunks = ([], inputs)
            self.labels = (labels, [])
        self.criterion = criterion

        # For the first half of the ranks: phase 0 means forward direction, phase 1 means reverse direction.
        # For the second half of the ranks: phase 0 means reverse direction, phase 1 means forward direction.

        # Step 1: nF0
        ## 1. nF0: Forward computations for phase 0 (warm-up).
        step_1 = (num_half_ranks - half_rank - 1) * 2
        for i in range(step_1):
            self._forward_chunk(0)

        # Step 2: nF0F1
        ## 2. nF0F1: Alternates forward for phases 0 and 1 with communication.
        step_2 = half_rank + 1
        self._recv_forward(0)
        for i in range(step_2):
            self._forward_chunk(0, recv=False, send=self.is_middle_rank)
            self._recv_forward(0)
            self._forward_chunk(1, send=(not self.is_middle_rank) or (i < step_2 - 1))
            if not self.is_middle_rank:
                self._send_forward(0)

        # Step 3: nB1W1F1 (Use zero bubble)
        ## 3. nB1W1F1: Backward for phase 1, weight updates, and forward for phase 1 with zero-bubble optimization.
        step_3 = num_half_ranks - half_rank - 1
        for i in range(step_3):
            self._backward_chunk(1, enable_zb=True)
            self._recv_forward(1)
            self._weight_chunk()
            self._forward_chunk(1, recv=False)

        # Step 4 (Main step): nF0B1F1B0
        ## 4. nF0B1F1B0: Main overlap phase for forward and backward in both directions.
        step_4 = half_num_chunks - num_ranks + half_rank + 1
        for i in range(step_4):
            if i == 0:
                if self.is_middle_rank:
                    # NOTE: We don't overlap these two chunks to further reduce bubble size.
                    self._forward_chunk(0, recv=False, send=False)
                    self._send_forward(1)
                    self._backward_chunk(1, send=False)
                    self._send_forward(0)
                    self._send_backward(1)
                else:
                    self._forward_backward_chunk(0, 1, recv0=False)
            else:
                self._forward_backward_chunk(0, 1)
            self._forward_backward_chunk(1, 0)

        # Step 5: nB1F1B0
        ## 5. nB1F1B0: Backward for phase 1 with overlapped forward-backward for phase 0.
        step_5 = num_half_ranks - half_rank - 1
        for i in range(step_5):
            self._backward_chunk(1)
            self._forward_backward_chunk(1, 0)

        # Step 6: nB1B0 (The second half of the chunks use zero bubble)
        ## 6. nB1B0: Backward for both phases, with zero-bubble for the second half.
        step_6 = half_rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and half_rank % 2 == 1:
                enable_zb = True
            self._backward_chunk(1, enable_zb=enable_zb)
            if i == step_6 // 2 and half_rank % 2 == 0:
                enable_zb = True
            self._backward_chunk(0, enable_zb=enable_zb)

        # Step 7: nWB0 (Use zero bubble)
        ## 7. nWB0: Weight updates and backward for phase 0 with zero-bubble.
        step_7 = num_half_ranks - half_rank - 1
        for i in range(step_7):
            self._weight_chunk()
            self._backward_chunk(0, enable_zb=True)

        # Step 8: nW
        ## 8. nW: Final weight updates.
        step_8 = half_rank + 1
        for i in range(step_8):
            self._weight_chunk()
        assert WeightGradStore.funcs_queue.empty()

        self._commit_and_wait_comm()

        loss, outputs = None, None
        if self.is_first_rank or self.is_last_rank:
            if criterion is not None:
                loss = torch.stack(self.loss_chunks)
            if return_outputs:
                outputs = gather(self.output_chunks[self.is_first_rank], self.batch_dim)
                if len(outputs) == 1:
                    outputs = outputs[0]

        self._reset_states()

        return loss, outputs
