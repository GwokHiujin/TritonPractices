import torch
import triton
import triton.language as tl
import triton.testing as testing


@triton.jit
def min_kernel(input_ptr, 
               # Matrix dimensions
               M, 
               N,
               stride_x, 
               stride_y, 
               output_ptr, 
               dim, 
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=(1 if dim == 0 else 0))

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr, 
        shape=(N if dim == 0 else M,), 
        strides=(1,), 
        offsets=(pid,), 
        block_shape=(1,), 
        order=(0,)
    )
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr, 
        shape=(M, N),
        strides=(stride_x, stride_y),
        offsets=(0, pid) if dim == 0 else (pid, 0),
        block_shape=(BLOCK_SIZE, 1) if dim == 0 else (1, BLOCK_SIZE), 
        order=(1, 0)
    )
    input = tl.load(input_block_ptr, boundary_check=(0, 1))
    output = tl.min(input, axis=(0 if dim == 0 else 1))
    tl.store(output_block_ptr, output.to(tl.float32))


def _min(x, dim):
    M, N = x.shape
    if (dim == 0):
        # 按列输出结果
        BLOCK_SIZE = triton.next_power_of_2(M)
    elif (dim == 1):
        # 按行输出结果
        BLOCK_SIZE = triton.next_power_of_2(N)
    else:
        raise RuntimeError("ERROR: illegal dim(should be 0 or 1)")
    stride_x = x.stride(0)
    stride_y = x.stride(1)
    output = torch.empty(N if dim==0 else M, device='cuda',dtype=torch.float32)
    assert x.is_cuda and output.is_cuda

    def grid(meta):
        return (1, N) if dim==0 else (M,)

    min_kernel[grid](x, M, N, stride_x, stride_y, output, dim=dim, BLOCK_SIZE=BLOCK_SIZE)
    return output


# TEST CODE
torch.manual_seed(0)
N = (8, 4)
x = torch.rand(N, device='cuda')
output_torch_dim0 = torch.min(x, dim=0)[0]
output_torch_dim1 = torch.min(x, dim=1)[0]
output_triton_dim0 = _min(x, dim=0)
output_triton_dim1 = _min(x, dim=1)
print(f'Origin Tensor: {x}')
print(f'Torch output(dim=0): {output_torch_dim0}')
print(f'Torch output(dim=1): {output_torch_dim1}')
print(f'Triton output(dim=0): {output_triton_dim0}')
print(f'Triton output(dim=1): {output_triton_dim1}')
print(f"The output of torch and triton(dim0) is {'✅SAME' if torch.allclose(output_torch_dim0, output_triton_dim0) else '🚨DIFF'}")
print(f"The output of torch and triton(dim1) is {'✅SAME' if torch.allclose(output_torch_dim1, output_triton_dim1) else '🚨DIFF'}")