import triton
import triton.language as tl
import triton.testing as testing
import torch
import random


base = 5


# Clamps all elements in input into the given range [ min, max ]. 
# If min is None, there is no lower bound. 
# Or, if max is None there is no upper bound.
# If min is greater than max, sets all elements in input to the value of max.

# It means that: 
#    we need to replace all the input's elements which is less than min to min, 
#    which is greater than max to max 

# Parameters: input, min(optional), max(optional)
@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": m})
        for m in [16, 32, 64, 128, 256, 512]
    ],
    key=['n_elements']
)
@triton.jit
def clamp_kernel(input_ptr, 
                 n_elements, 
                 output_ptr, 
                 min, 
                 max, 
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE

    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(input < min, min, input)
    output = tl.where(output > max, max, output)

    tl.store(output_ptr + offsets, value=output, mask=mask)


def _clamp(x: torch.Tensor, min=float('-inf'), max=float('inf')):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda

    if (min == None and max == None):
        raise RuntimeError("At least one of 'min' or 'max' must not be None")

    if (min == None):
        min = float('-inf')
    elif (max == None):
        max = float('inf')

    if (min > max):
        output = max
        return output

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    clamp_kernel[grid](x, n_elements, output, min, max)
    return output


@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["N"],
            x_vals=[16 * i for i in range(1, 16, 1)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            ylabel="milliseconds",
            plot_name="06-clamp-performance",
            args={"M": 64},
        ),
    ]
)
def benchmark(M, N, backend):
    input_size = (M, N)
    input = torch.rand(input_size, device='cuda') + base
    min = random.random() + base
    max = random.random() + base

    if backend == "triton":
        return testing.do_bench(lambda: _clamp(input, min=min, max=max))
    else:
        return testing.do_bench(lambda: torch.clamp(input, min=min, max=max))


# TEST CODE
torch.manual_seed(0)
x = torch.rand((2, 4), device='cuda') + base
min = random.choice([random.random() + base, None])
max = random.choice([random.random() + base, None])
output_torch = torch.clamp(x, min, max)
output_triton = _clamp(x, min, max)
print(f'Origin Tensor x: {x}')
print(f'min: {min}, max: {max}')
print(f'Torch output: {output_torch}')
print(f'Triton output: {output_triton}')
print(f"The output of torch and triton is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")
print(f'BENCHMARKING')
benchmark.run(show_plots=True, print_data=True, save_path='./benchmark-results/')
print(f'Successfully run the benchmark')