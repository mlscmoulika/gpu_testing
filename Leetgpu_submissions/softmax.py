#  https://leetgpu.com/challenges/softmax - Medium question :))

import torch
import triton
import triton.language as tl
import time


@triton.jit
def softmax_kernel(input, output, N, BLOCK_SIZE: tl.constexpr, ma, deno):
    pid =tl.program_id(0)
    chunk = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = chunk < N
    i = tl.load(input + chunk, mask=mask, other=0.0)
    
    a = tl.exp(i-ma)/deno
    tl.store(output + chunk, a, mask=mask)


# input, output are tensors on the GPU

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE=1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    ma = torch.max(input).item()
    deno = torch.sum(torch.exp(input - ma)).item()
    softmax_kernel[grid](input,output, N, BLOCK_SIZE, ma, deno)

N = 8192
print("Matrix addition of two matrices of size ", N, "x", N)
A = torch.randn(N, device="cuda")
# B = torch.randn(N, N, device="cuda")
C = torch.empty(N, device="cuda")
torch.cuda.synchronize()
t = time.time()
solve(A, C, N)   
torch.cuda.synchronize()
print("Time taken for matrix addition on GPU using Triton: ", time.time() - t)
# solving using torch
torch.cuda.synchronize()
t = time.time()
C_torch = torch.softmax(A, dim=0) 
torch.cuda.synchronize()
print("Time taken for matrix addition on GPU using PyTorch: ", time.time() - t)

# N = 1024
# print("Matrix addition of two matrices of size ", N, "x", N)
# A = torch.randn(N, 1, device="cuda")
# #B = torch.randn(N, N, device="cuda")
# C = torch.empty(N, 1, device="cuda")
# torch.cuda.synchronize()
# t = time.time()
# solve(A, C, N)   
# torch.cuda.synchronize()
# print("Time taken for matrix addition on GPU using Triton: ", time.time() - t)
# # solving using torch
# torch.cuda.synchronize()
# t = time.time()
# C_torch = torch.softmax(A, dim=1)
# torch.cuda.synchronize()
# print("Time taken for matrix addition on GPU using PyTorch: ", time.time() - t)

# N = 512
# print("Matrix addition of two matrices of size ", N, "x", N)
# A = torch.randn(N, 1, device="cuda")
# # B = torch.randn(N, N, device="cuda")
# C = torch.empty(N, 1, device="cuda")
# torch.cuda.synchronize()
# t = time.time()
# solve(A, C, N)   
# torch.cuda.synchronize()
# print("Time taken for matrix addition on GPU using Triton: ", time.time() - t)
# # solving using torch
# torch.cuda.synchronize()
# t = time.time()
# C_torch = torch.softmax(A, dim=1) 
# torch.cuda.synchronize()
# print("Time taken for matrix addition on GPU using PyTorch: ", time.time() - t)
