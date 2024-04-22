import triton
import triton.language as tl
import triton.testing as testing
import torch


threshlod = 0.8


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
    N = x.numel()
    BLOCK_SIZE = 2048
    # BLOCK_SIZE = triton.next_power_of_2(N)
    o_elements = N // BLOCK_SIZE
    output = torch.empty(o_elements, device='cuda')
    
    assert x.is_cuda and output.is_cuda
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    all_kernel[grid](x, output, N, BLOCK_SIZE=BLOCK_SIZE) # type: ignore
    return output.min() != 0
    

@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * i for i in range(1, 16, 1)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            ylabel="milliseconds",
            plot_name="01-naive-all-performance",
            args={"M": 8},
        ),
    ]
)
def benchmark(M, N, backend):
    input = torch.rand(M, N, device="cuda") < threshlod

    if backend == "triton":
        return testing.do_bench(lambda: _all(input))
    else:
        return testing.do_bench(lambda: torch.all(input))
    

# TEST CODE
torch.manual_seed(0)
N = 98432
x = torch.rand(N, device='cuda') < threshlod
output_torch = torch.all(x)
output_triton = _all(x)
print(f'Origin Tensor: {x}')
print(f'Torch output: {output_torch}')
print(f'Triton output: {output_triton}')
print(f"The output of torch and triton is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")
print(f'BENCHMARKING')
benchmark.run(show_plots=True, print_data=True, save_path='./benchmark-results/')
print(f'Successfully run the benchmark')