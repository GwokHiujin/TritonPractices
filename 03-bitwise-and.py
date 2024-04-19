import triton
import triton.language as tl
import torch

@triton.jit
def bitwise_and_kernel(x_ptr, 
               y_ptr, 
               output_ptr, 
               n_elements, 
               BLOCK_SIZE: tl.constexpr,):
    pid = tl.program_id(axis=0)     # 1D launch grid
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE   # Its size is equal to BLOCK_SIZE(a "block" of pointer)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)     # Should be a 1D "block"
    y = tl.load(y_ptr + offsets, mask=mask)     # Should be a 1D "block"
    output = x & y
    
    tl.store(output_ptr + offsets, value=output, mask=mask)


def bitwise_and(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    bitwise_and_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


torch.manual_seed(0)
size = 98264
x = torch.randint(0, 999, (2, 4), device='cuda')
y = torch.randint(0, 999, (2, 4), device='cuda')
output_torch = torch.bitwise_and(x, y)
output_triton = bitwise_and(x, y)
print(f'Origin Tensor x: {x}')
print(f'Origin Tensor y: {y}')
print(f'Torch output: {output_torch}')
print(f'Triton output: {output_triton}')
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
