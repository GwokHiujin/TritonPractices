import triton
import triton.language as tl
import torch


# Tests if all elements in input evaluate to True.
# naive-all only support 1-dimension tensor(i.e. The parameter 'dim' is unsupported)
@triton.jit
def all_kernel(input_ptr, 
               output_ptr, 
               n_elements, 
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    input_offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = input_offsets < n_elements

    input = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + pid, tl.min(input, axis=0) == 1)


def _all(x: torch.Tensor):
    size = x.numel()
    BLOCK_SIZE = 1024
    # BLOCK_SIZE = triton.next_power_of_2(size)
    o_elements = size // BLOCK_SIZE
    output = torch.empty(o_elements, device='cuda')
    assert x.is_cuda and output.is_cuda
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    all_kernel[grid](x, output, size, BLOCK_SIZE=BLOCK_SIZE) # type: ignore
    return output.min() != 0
    

# TEST CODE
torch.manual_seed(0)
threshlod = 0.5
size = 98432
x = torch.rand(size, device='cuda') < threshlod
# x = torch.ones(size, dtype=torch.bool, device='cuda') # All True test case 
output_torch = torch.all(x)
output_triton = _all(x)
print(f'Origin Tensor: {x}')
print(f'Torch output: {output_torch}')
print(f'Triton output: {output_triton}')
print(f"The output of torch and triton is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")
