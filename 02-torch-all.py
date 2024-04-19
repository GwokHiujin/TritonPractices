import triton
import triton.language as tl
import torch


# Tests if any element in input evaluate to True.
# torch-all can support multi-dimensions tensor
@triton.jit
def all_kernel(input_ptr, 
               # Matrix dimensions
               M, 
               N, 
               stride_m, 
               stride_n,
               output_ptr, 
               BLOCK_SIZE_M: tl.constexpr, 
               BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)


def _all(x: torch.Tensor, dim):
    M, N = x.shape


# TEST CODE
torch.manual_seed(0)
threshlod = 0.5
size = (32, 16)
x = torch.rand(size, device='cuda') < threshlod
# x = torch.ones(size, dtype=torch.bool, device='cuda') # All True test case 
output_torch_dim0 = torch.all(x, dim=0)
output_torch_dim1 = torch.all(x, dim=1)
print(f'Origin Tensor: {x}')
print(f'Torch output(dim=0): {output_torch_dim0}')
print(f'Torch output(dim=1): {output_torch_dim1}')