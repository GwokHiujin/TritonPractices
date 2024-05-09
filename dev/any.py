import torch
import triton
import math
import triton.language as tl


# torch.any: Tests if any elements in input evaluate to True.
#            If the dtype of input is not BOOL, then test if any elements in input evaluate to non-zero value
# In triton function, test if any elements in input evaluate to non-zero value is ok. 
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=5),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    values={"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])},
)
@triton.jit
def any_kernel_dim(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    inp_offset = m_offset[:, None, None] * N * K + n_offset[None, :, None] * K + pid_k
    inp_mask = (m_offset[:, None, None] < M) & (n_offset[None, :, None] < N)
    out_offset = m_offset[:, None] * K + pid_k
    out_mask = (m_offset[:, None] < M)

    inp_ptrs = inp + inp_offset
    inp_vals = tl.load(inp_ptrs, mask=inp_mask, other=0.0).to(tl.float32)
    result = tl.min(inp_vals == 0, axis=1)

    out_ptrs = out + out_offset
    tl.store(out_ptrs, result, mask=out_mask)

@triton.jit
def any_kernel_tensor_1(
    inp,
    mid,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)
    any_val = tl.min(inp_val == 0., axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, any_val)

@triton.jit
def any_kernel_tensor_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    any_val = tl.min(mid_val == 0, axis=0)
    tl.store(out, any_val)


def any(inp, dim=None, keepdim=False, *, dtype=None):
    if __debug__:
        print("GEMS any")
    assert (dim == None) or (dim >= -inp.ndim and dim < inp.ndim), "Invalid dim"

    if dtype is None:
        dtype = inp.dtype

    if (dim == None):
        n_elements = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        mid_size = triton.cdiv(n_elements, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)

        any_kernel_tensor_1[(mid_size, 1, 1)](inp, mid, n_elements, block_size)
        any_kernel_tensor_2[(1, 1, 1)](mid, out, mid_size, block_mid)

        return (out != 0.)

    shape = list(inp.shape)
    dim = dim % len(shape)
    M = 1
    N = shape[dim]
    del shape[dim]
    for i in range(dim):
        M *= shape[i]
    if (keepdim):
        shape.insert(dim, 1)
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
    out = torch.empty(tuple(shape), dtype=dtype, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    print(f'M: {M}, N: {N}, K: {K}')
    any_kernel_dim[grid](inp, out, M, N, K)
    return (out == 0.)


# TEST CODE
x = torch.rand((2), device='cuda') < 0.5
# x = torch.rand((2, 3, 4, 5), device='cuda')
dim = 2
keepdim = False
print(f'x: {x}')

out_torch = torch.any(x)
print(f'torch out: {out_torch}')
out_triton = any(x)
print(f'triton out: {out_triton}')

print(f"The output of torch and triton is {'✅SAME' if torch.allclose(out_torch, out_triton) else '🚨DIFF'}")