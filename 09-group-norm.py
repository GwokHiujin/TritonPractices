import triton
import triton.language as tl
import triton.testing as testing
import torch


@triton.jit
def group_norm_kernel(input_ptr, 
                      output_ptr, 
                      M, 
                      N, 
                      ## parameters of the group-norm function
                      gamma, 
                      beta, 
                      mean, 
                      stdv, # standard-deviation
                      eps, 
                      BLOCK_SIZE_M: tl.constexpr, 
                      BLOCK_SIZE_N: tl.constexpr
                      ):
    pid = tl.program_id(axis=0)

    # compute mean
    # TODO

    # compute standard-deviation
    # TODO


def _group_norm(input: torch.Tensor, 
                num_groups: int, 
                num_channels: int, 
                eps: float, 
                affine: bool):
    M, N = input.shape
    # TODO