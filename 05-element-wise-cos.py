import triton
import triton.language as tl
import triton.testing as testing
import torch


base = 5


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": m})
        for m in [32, 64, 128, 256, 512]
    ],
    key=['n_elements']
)
@triton.jit
def cos_kernel(input_ptr, 
               output_ptr, 
               n_elements, 
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)     # 1D launch grid
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE   # Its size is equal to BLOCK_SIZE(a "block" of pointer)

    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)     # Should be a 1D "block"
    output = tl.cos(input)
    
    tl.store(output_ptr + offsets, value=output, mask=mask)


def _cos(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    cos_kernel[grid](x, output, n_elements)
    return output


@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["N"],
            x_vals=[128 * i for i in range(1, 16, 1)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            ylabel="milliseconds",
            plot_name="05-cos-performance",
            args={"M": 8},
        ),
    ]
)
def benchmark(M, N, backend):
    input_size = (M, N)
    input = torch.rand(input_size, device='cuda') + base

    if backend == "triton":
        return testing.do_bench(lambda: _cos(input))
    else:
        return testing.do_bench(lambda: torch.cos(input))


# TEST CODE
torch.manual_seed(0)
x = torch.rand((2, 4), device='cuda') + base
output_torch = torch.cos(x)
output_triton = _cos(x)
print(f'Origin Tensor x: {x}')
print(f'Torch output: {output_torch}')
print(f'Triton output: {output_triton}')
print(f"The output of torch and triton is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")
print(f'BENCHMARKING')
benchmark.run(show_plots=True, print_data=True, save_path='./benchmark-results/')
print(f'Successfully run the benchmark')