import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    r = pid_0*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
    c = pid_1*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)

    a = tl.load(A + r[:, None]*N+c[None, :], mask=(r[:,None]<N)&(c[None, :]<N))
    b = tl.load(B+ r[:, None]*N+c[None, :], mask=(r[:,None]<N)&(c[None, :]<N))

    c_ans=a+b 
    tl.store(C+r[:, None]*N+c[None, :], c_ans, mask=(r[:,None]<N)&(c[None, :]<N))
    


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),triton.cdiv(N, BLOCK_SIZE))
    matrix_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
