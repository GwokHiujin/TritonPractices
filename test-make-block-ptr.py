import triton
import triton.language as tl
import torch


def sum_row_blocked(A: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    outputs = torch.empty((M,), dtype=A.dtype, device=A.device)

    dynamic_launch_grid = lambda params: (triton.cdiv(M, params["BLOCK_M"]), )
    sum_row_blocked_kernel[dynamic_launch_grid](
        A_ptr=A, outputs_ptr=outputs,
        M=M, N=N,
        A_strides_x=A.stride(0), A_strides_y=A.stride(1),
        BLOCK_M=2,
    )

    return outputs


@triton.jit
def sum_row_blocked_kernel(
    A_ptr, outputs_ptr,
    M, N,
    BLOCK_M,
    A_strides_x, A_strides_y,
):
    program_id = tl.program_id(axis=0)
    input_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, N),
        strides=(A_strides_x, A_strides_y),
        offsets=(program_id * BLOCK_M, 0),
        block_shape=(BLOCK_M, N),
        order=(1, 0),
    )


# TEST CODE
torch.manual_seed(0)
N = (8, 16)
x = torch.rand(N, device='cuda')
print(f'x\'s stride-x is: {x.stride(0)}, x\'s stride-y is {x.stride(1)}')