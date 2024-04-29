import triton
import triton.language as tl
import triton.testing as testing
import torch


base = 5

# the implement of dim = 1
@triton.jit
def log_softmax_kernel(x_ptr, z_ptr, N0, N1, T, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_0 = tl.program_id(0)

    offsets_x = tl.arange(0, BLOCK_SIZE_M) + pid_0 * BLOCK_SIZE_M
    mask_x = offsets_x < N0

    # calculate x_max and exp_x_sum
    # x_max's size should be (N0, )
    x_max = tl.full((BLOCK_SIZE_M, ), float('-inf'), dtype=tl.float32)
    exp_x_sum = tl.full((BLOCK_SIZE_M, ), 0.0, dtype=tl.float32)
    for i in tl.range(0, T, BLOCK_SIZE_N):
      offsets_y = tl.arange(0, BLOCK_SIZE_N) + i
      offsets = offsets_x[:, None] * T + offsets_y[None, :]
      mask = (offsets_x[:, None] < N0) & (offsets_y[None, :] < T)

      x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
      tmp_max = tl.maximum(tl.max(x, axis=1), x_max)
      # Use the Hint:
      # exp(x - newMax) = exp(x - oldMax) * exp(oldMax - newMax) -> Call it a Delta, or Correction Value
      # So the SUM in every loop will be [currentValue + Delta]
      # if you dont use this Hint, you will need a new loop to calculate the exp_x_sum
      exp_x_sum = tl.sum(_exp(x - tmp_max[:, None]), axis=1) + exp_x_sum * _exp(x_max - tmp_max)
      x_max = tmp_max

    # calculate log_softmax result
    for i in tl.range(0, T, BLOCK_SIZE_N):
      offsets_y = tl.arange(0, BLOCK_SIZE_N) + i
      offsets = offsets_x[:, None] * T + offsets_y[None, :]
      mask = (offsets_x[:, None] < N0) & (offsets_y[None, :] < T)

      x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
      exp_x = _exp(x - x_max[:, None])
      output = exp_x / exp_x_sum[:, None]

      tl.store(z_ptr + offsets, tl.log(output), mask=mask)
    return

def _exp(x):
    log2_e = 1.44269504
    return tl.exp2(x * log2_e)


def _log_softmax(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    M, N = x.shape
    n_elements = x.numel()

    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 32
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), 
                         N, meta['BLOCK_SIZE_N'])
    log_softmax_kernel[grid](x, output, M, n_elements, N, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return output


# TEST CODE
torch.manual_seed(0)
x = torch.rand((2, 4), device='cuda') + base
output_torch = torch.log_softmax(x, dim=1)
output_triton = _log_softmax(x)
print(f'Origin Tensor x: {x}')
print(f'Torch output: {output_torch}')
print(f'Triton output: {output_triton}')
print(f"The output of torch and triton is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")

  