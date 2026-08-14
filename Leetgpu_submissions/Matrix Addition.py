import torch
import triton
import triton.language as tl
import time


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
    BLOCK_SIZE = 512
    
    grid = (triton.cdiv(N, BLOCK_SIZE),triton.cdiv(N, BLOCK_SIZE))
    matrix_add_kernel[grid](a, b, c, N, BLOCK_SIZE)

N = 512
print("Matrix addition of two matrices of size ", N, "x", N)
A = torch.randn(N, N, device="cuda")
B = torch.randn(N, N, device="cuda")
C = torch.empty(N, N, device="cuda")
torch.cuda.synchronize()
t = time.time()
solve(A, B, C, N)   
torch.cuda.synchronize()
print("Time taken for matrix addition on GPU using Triton: ", time.time() - t)
# solving using torch
t = time.time()
C_torch = A + B 
torch.cuda.synchronize()
print("Time taken for matrix addition on GPU using PyTorch: ", time.time() - t)

# N = 8192
# print("Matrix addition of two matrices of size ", N, "x", N)
# A = torch.randn(N, N, device="cuda")
# B = torch.randn(N, N, device="cuda")
# C = torch.empty(N, N, device="cuda")
# torch.cuda.synchronize()
# t = time.time()
# solve(A, B, C, N)   
# torch.cuda.synchronize()
# print("Time taken for matrix addition on GPU using Triton: ", time.time() - t)
# # solving using torch
# torch.cuda.synchronize()
# t = time.time()
# C_torch = A + B 
# torch.cuda.synchronize()
# print("Time taken for matrix addition on GPU using PyTorch: ", time.time() - t)

# N = 1024
# print("Matrix addition of two matrices of size ", N, "x", N)
# A = torch.randn(N, N, device="cuda")
# B = torch.randn(N, N, device="cuda")
# C = torch.empty(N, N, device="cuda")
# torch.cuda.synchronize()
# t = time.time()
# solve(A, B, C, N)   
# torch.cuda.synchronize()
# print("Time taken for matrix addition on GPU using Triton: ", time.time() - t)
# # solving using torch
# torch.cuda.synchronize()
# t = time.time()
# C_torch = A + B 
# torch.cuda.synchronize()
# print("Time taken for matrix addition on GPU using PyTorch: ", time.time() - t)
