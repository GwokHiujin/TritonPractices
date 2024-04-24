import triton
import triton.language as tl
import triton.testing as testing
import torch
import random
import math

#Computes a vector norm.
# If x is complex valued, it computes the norm of x.abs()
# if ord==inf, calculate max(abs(x))
# if ord==-inf, calculate min(abs(x))
# if ord==0, calculate sum(x != 0)
# if ord==other int or float, calculate sum(abs(x)^{ord})^{(1 / ord)}
@triton.jit
def vector_norm_kernel(input_ptr, 
               output_ptr, 
               n_elements, 
               ord, 
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    input_offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = input_offsets < n_elements

    input = tl.load(input_ptr + input_offsets, mask=mask)

    if (ord == float('inf')):
        output = tl.max(tl.abs(input))
    elif (ord == -float('inf')):
        output = tl.min(tl.abs(input))
    else:
        output = tl.sum(tl.exp(tl.log(tl.abs(input)) * ord))
    
    tl.store(output_ptr + pid, output.to(tl.float32))


def _vector_norm(x: torch.Tensor, ord=2):
    N = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(N)
    output = torch.empty(1, device='cuda')

    if (ord == 0):
        return torch.sum(x != 0).to(torch.float32)
    
    assert x.is_cuda and output.is_cuda
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    vector_norm_kernel[grid](x, output, N, ord, BLOCK_SIZE=BLOCK_SIZE)
    if (ord != float('inf') and 
        ord != float('-inf')):
        return output ** (1 / ord)
    else:
        return output
    

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
            plot_name="08-vector-norm-performance",
            args={},
        ),
    ]
)
def benchmark(N, backend):
    input = torch.rand(N, device="cuda")
    ord = random.random() + random.randint(0, 5)

    if backend == "triton":
        return testing.do_bench(lambda: _vector_norm(input, ord=ord))
    else:
        return testing.do_bench(lambda: torch.linalg.norm(input, ord=ord))
    

# TEST CODE
torch.manual_seed(0)
N = 1024
x = torch.rand(N, device='cuda')
ord = random.choice([float('inf'), 
                     float('-inf'), 
                     0, 
                     random.randint(2, 16), 
                     random.random() + random.randint(0, 5)]
                     )
output_torch = torch.linalg.norm(x, ord=ord)
output_triton = _vector_norm(x, ord=ord)
print(f'Origin Tensor: {x}')
print(f'Ord: {ord}')
print(f'Torch output: {output_torch}')
print(f'Triton output: {output_triton.item()}')
print(f"The output of torch and triton is {'✅SAME' if torch.allclose(output_torch, output_triton) else '🚨DIFF'}")
print(f'BENCHMARKING')
benchmark.run(show_plots=True, print_data=True, save_path='./benchmark-results/')
print(f'Successfully run the benchmark')