import triton
import triton.language as tl
import triton.testing as testing
import torch


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": m})
        for m in [32, 64, 128, 256, 512]
    ],
    key=['n_elements']
)
@triton.jit
def bitwise_and_kernel(x_ptr, 
               y_ptr, 
               output_ptr, 
               n_elements, 
               BLOCK_SIZE: tl.constexpr):
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
    bitwise_and_kernel[grid](x, y, output, n_elements)
    return output


@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["size_x"],
            x_vals=[128 * i for i in range(1, 16, 1)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            ylabel="milliseconds",
            plot_name="03-bitwise-and-performance",
            args={"num_batches": 8},
        ),
    ]
)
def benchmark(num_batches, size_x, backend):
    input_size = (size_x, size_x // 4)
    input_x = torch.randint(0, 999, input_size, device='cuda')
    input_y = torch.randint(0, 999, input_size, device='cuda')

    if backend == "triton":
        return testing.do_bench(lambda: bitwise_and(input_x, input_y))
    else:
        return testing.do_bench(lambda: torch.bitwise_and(input_x, input_y))


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
print(f"The output of torch and triton is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")
print(f'BENCHMARKING')
benchmark.run(show_plots=True, print_data=True, save_path='./benchmark-results/')
print(f'Successfully run the benchmark')