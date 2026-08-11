#  https://leetgpu.com/challenges/softmax - Medium question :))

import torch
import triton
import triton.language as tl


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
