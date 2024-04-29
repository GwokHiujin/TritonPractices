import triton
import triton.language as tl
import triton.testing as testing
import torch

# Returns the sum of all elements in the input tensor.
# 感觉跟 all 一样，还是得指定 shape，每个 shape 不一样的算子，一个算子管不了所有 shape
# >>> a = torch.randn(4, 4)
# >>> a
# tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
#         [-0.2993,  0.9138,  0.9337, -1.6864],
#         [ 0.1132,  0.7892, -0.1003,  0.5688],
#         [ 0.3637, -0.9906, -0.4752, -1.5197]])
# >>> torch.sum(a, 1)
# tensor([-0.4598, -0.1381,  1.3708, -2.6217])
@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets_x = tl.arange(0, BLOCK_SIZE_M) + pid * BLOCK_SIZE_M
    mask_x = offsets_x < N0

    output = tl.full((BLOCK_SIZE_M, ), 0., dtype=tl.float32)
    for i in tl.range(0, T, BLOCK_SIZE_N):
        offsets_y = tl.arange(0, BLOCK_SIZE_N) + i
        offsets = offsets_x[:, None] * T + offsets_y[None, :]
        mask = (offsets_x[:, None] < N0) & (offsets_y[None, :] < T)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        output += tl.sum(x, axis=1)
    tl.store(z_ptr + offsets_x, output, mask=mask_x)


def _sum(x: torch.Tensor):
    M, N = x.shape
    stride_x = x.stride(0)
    stride_y = x.stride(1)
    output = torch.empty(M, device='cuda',dtype=torch.float32)
    assert x.is_cuda and output.is_cuda

    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 32
    n_elements = x.numel()

    def grid(meta):
        return (M,)
    
    sum_kernel[grid](x, output, M, n_elements, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return output


# TEST CODE
torch.manual_seed(0)
N = (8, 4)
x = torch.rand(N, device='cuda')
output_torch = torch.sum(x)
output_triton = _sum(x)
print(f'Origin Tensor: {x}')
print(f'Torch output(dim=0): {output_torch}')
print(f'Triton output(dim=0): {output_triton}')
print(f"The output of torch and triton(dim0) is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")