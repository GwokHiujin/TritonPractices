import triton
import triton.language as tl
import triton.testing as testing
import torch
import random


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE_0": m, "BLOCK_SIZE_1": n})
        for m in [2, 4, 8]
        for n in [2, 4, 8]
    ],
    key=['input_size']
)
@triton.jit
def outer_kernel(input_ptr, 
                 vec2_ptr, 
                 # Vector size
                 input_size, 
                 vec2_size, 
                 output_ptr,
                 BLOCK_SIZE_0: tl.constexpr, 
                 BLOCK_SIZE_1: tl.constexpr):
    pid_0 = tl.program_id(axis=0)
    pid_1 = tl.program_id(axis=1)

    input_offsets = tl.arange(0, BLOCK_SIZE_0) + pid_0 * BLOCK_SIZE_0
    vec2_offsets = tl.arange(0, BLOCK_SIZE_1) + pid_1 * BLOCK_SIZE_1
    output_offsets = vec2_offsets[:, None] + input_offsets[None, :] * vec2_size
    mask0 = input_offsets < input_size
    mask1 = vec2_offsets < vec2_size
    mask2 = (input_offsets[None, :] < input_size) and (vec2_offsets[:, None] < vec2_size)

    input = tl.load(input_ptr + input_offsets, mask=mask0)
    vec2 = tl.load(vec2_ptr + vec2_offsets, mask=mask1)
    output = input[None, :] * vec2[:, None]

    tl.store(output_ptr + output_offsets, output, mask=mask2)
    

def _outer(x: torch.Tensor, y: torch.Tensor):
    input_size = x.numel()
    vec2_size = y.numel()
    assert x.dim() == 1 and y.dim() == 1

    output = torch.empty((input_size, vec2_size), device='cuda')
    assert x.is_cuda and y.is_cuda and output.is_cuda

    grid = lambda meta: (triton.cdiv(input_size, meta['BLOCK_SIZE_0']), 
                         triton.cdiv(vec2_size, meta['BLOCK_SIZE_1']))
    outer_kernel[grid](x, y, input_size, vec2_size, output)
    return output


@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["vector_size"],
            x_vals=[8 * i for i in range(1, 16, 1)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            ylabel="milliseconds",
            plot_name="07-outer-performance", 
            args={}
        ),
    ]
)
def benchmark(vector_size, backend):
    input = torch.rand(vector_size, device='cuda')
    vec2 = torch.rand(vector_size, device='cuda')

    if backend == "triton":
        return testing.do_bench(lambda: _outer(input, vec2))
    else:
        return testing.do_bench(lambda: torch.outer(input, vec2))


# TEST CODE
torch.manual_seed(0)
M = random.randint(8, 64)
N = random.randint(8, 64)
x = torch.rand(M, device='cuda')
y = torch.rand(N, device='cuda')
print(f'Origin Tensor x: {x}')
print(f'Origin Tensor y: {y}')

output_torch = torch.outer(x, y)
output_triton = _outer(x, y)
print(f'Torch output: {output_torch}')
print(f'Triton output: {output_triton}')
print(f"The output of torch and triton is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")
print(f'BENCHMARKING')
benchmark.run(show_plots=True, print_data=True, save_path='./benchmark-results/')
print(f'Successfully run the benchmark')