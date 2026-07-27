import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, BLOCK:tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    r = pid_r * BLOCK + tl.arange(0, BLOCK)
    c = pid_c * BLOCK + tl.arange(0, BLOCK)

    chunk = r[:, None]+c[None, :]

    i = tl.load(input+r[None, :]*cols+c[:, None], mask=(r[None, :]<rows)&(c[:, None]<cols), other=0.0)
    tl.store(output+r[None, :]+c[:, None]*rows, i, mask=(r[None, :]<rows)&(c[:, None]<cols))



# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    BLOCK = 64
    grid = (triton.cdiv(rows, BLOCK), triton.cdiv(cols, BLOCK),)
    matrix_transpose_kernel[grid](
        input, output, rows, cols, BLOCK = BLOCK)
