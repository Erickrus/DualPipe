from typing import List, Tuple

import torch
import torch.distributed as dist

'''
Logic: 
    Uses PyTorch’s dist.P2POp for asynchronous point-to-point communication.
'''

## Predefined shapes and data type for communication tensors, set via set_p2p_tensor_shapes and set_p2p_tensor_dtype.
TENSOR_SHAPES: List[Tuple[int]] = None
TENSOR_DTYPE: torch.dtype = None

def set_p2p_tensor_shapes(shapes: List[Tuple[int]]):
    global TENSOR_SHAPES
    TENSOR_SHAPES = shapes


def set_p2p_tensor_dtype(dtype: torch.dtype):
    global TENSOR_DTYPE
    TENSOR_DTYPE = dtype


def build_from_tensor_shapes():
    return [torch.empty(s, dtype=TENSOR_DTYPE, device="cuda", requires_grad=True) for s in TENSOR_SHAPES]


def append_irecv(ops: List[dist.P2POp], src: int, group: dist.ProcessGroup) -> List[torch.Tensor]:
    '''
    Queues a receive operation for input/gradient tensors.
    '''
    tensors = build_from_tensor_shapes()
    src = dist.distributed_c10d.get_global_rank(group, src)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.irecv, tensor, src))
    return tensors


def append_isend(ops: List[dist.P2POp], tensors: List[torch.Tensor], dst: int, group: dist.ProcessGroup) -> None:
    '''
    Queues a send operation for output/input gradient tensors.
    '''
    dst = dist.distributed_c10d.get_global_rank(group, dst)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.isend, tensor, dst))
