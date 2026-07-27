import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    ch = pid*BLOCK_SIZE+tl.arange(0, BLOCK_SIZE)
    i = tl.load(image + ch, mask=ch<height*width*4, other=0.0)
    i_ans = tl.where(ch%4==3, i, 255-i)
    tl.store(image+ch, i_ans, mask=ch<height*width*4)


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height * 4
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)
