import triton
import triton.language as tl
import triton.testing as testing
import torch


threshlod = 0.8


# Tests if any element in input evaluate to True.
# torch-all can support multi-dimensions tensor
# 还是得指定 shape，每个 shape 不一样的算子，一个算子管不了所有 shape
@triton.jit
def all_kernel(input_ptr, 
               # Matrix dimensions
               M, 
               N,
               stride_x, 
               stride_y, 
               output_ptr, 
               dim, 
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=(1 if dim == 0 else 0))

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr, 
        shape=(N if dim == 0 else M,), 
        strides=(1,), 
        offsets=(pid,), 
        block_shape=(1,), 
        order=(0,)
    )
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr, 
        shape=(M, N),
        strides=(stride_x, stride_y),
        offsets=(0, pid) if dim == 0 else (pid, 0),
        block_shape=(BLOCK_SIZE, 1) if dim == 0 else (1, BLOCK_SIZE), 
        order=(1, 0)
    )
    input = tl.load(input_block_ptr, boundary_check=(0, 1))
    output = tl.min(input, axis=(0 if dim == 0 else 1))
    tl.store(output_block_ptr, output.to(tl.int32))


def _all(x: torch.Tensor, dim):
    M, N = x.shape
    if (dim == 0):
        # 按列输出结果
        BLOCK_SIZE = triton.next_power_of_2(M)
    elif (dim == 1):
        # 按行输出结果
        BLOCK_SIZE = triton.next_power_of_2(N)
    else:
        raise RuntimeError("ERROR: illegal dim(should be 0 or 1)")
    stride_x = x.stride(0)
    stride_y = x.stride(1)
    output = torch.empty(N if dim==0 else M, device='cuda',dtype=torch.int32)
    assert x.is_cuda and output.is_cuda

    def grid(meta):
        return (1, N) if dim==0 else (M,)
    
    all_kernel[grid](x, M, N, stride_x, stride_y, output, dim=dim, BLOCK_SIZE=BLOCK_SIZE)
    return output == 1


@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["N"],
            x_vals=[8 * i for i in range(1, 16, 1)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            ylabel="milliseconds",
            plot_name="02-torch-all-performance-dim0",
            args={"M": 64},
        ),
    ]
)
def benchmark0(M, N, backend):
    input = torch.rand(M, N, device='cuda') < threshlod
    dim = 0

    if backend == "triton":
        return testing.do_bench(lambda: _all(input, dim=dim))
    else:
        return testing.do_bench(lambda: torch.all(input, dim=dim))
    

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
            plot_name="02-torch-all-performance-dim1",
            args={"M": 32},
        ),
    ]
)
def benchmark1(M, N, backend):
    input = torch.rand(M, N, device='cuda') < threshlod
    dim = 1

    if backend == "triton":
        return testing.do_bench(lambda: _all(input, dim=dim))
    else:
        return testing.do_bench(lambda: torch.all(input, dim=dim))


# TEST CODE
torch.manual_seed(0)
N = (8, 4)
x = torch.rand(N, device='cuda') < threshlod
output_torch_dim0 = torch.all(x, dim=0)
output_torch_dim1 = torch.all(x, dim=1)
output_triton_dim0 = _all(x, dim=0)
output_triton_dim1 = _all(x, dim=1)
print(f'Origin Tensor: {x}')
print(f'Torch output(dim=0): {output_torch_dim0}')
print(f'Torch output(dim=1): {output_torch_dim1}')
print(f'Triton output(dim=0): {output_triton_dim0}')
print(f'Triton output(dim=1): {output_triton_dim1}')
print(f"The output of torch and triton(dim0) is {'✅SAME' if torch.allclose(output_torch_dim0, output_triton_dim0) else '🚨DIFF'}")
print(f"The output of torch and triton(dim1) is {'✅SAME' if torch.allclose(output_torch_dim1, output_triton_dim1) else '🚨DIFF'}")
print(f'BENCHMARKING: dim0')
benchmark0.run(show_plots=True, print_data=True, save_path='./benchmark-results/')
print(f'BENCHMARKING: dim1')
benchmark1.run(show_plots=True, print_data=True, save_path='./benchmark-results/')
print(f'Successfully run the benchmark')