import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/column indices for this tile
    rows = pid_m * BLOCK + tl.arange(0, BLOCK)
    cols = pid_n * BLOCK + tl.arange(0, BLOCK)

    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    for k in range(0, K):
        a = tl.load(A + rows[:, None] * K + k,
                    mask=rows[:, None] < M,
                    other=0.0)

        b = tl.load(B + k * N + cols[None, :],
                    mask=cols[None, :] < N,
                    other=0.0)

        acc += a * b

    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


# Example
M = N = K = 128

A = torch.randn((M, K), device="cuda")
B = torch.randn((K, N), device="cuda")
C = torch.empty((M, N), device="cuda")

BLOCK = 16

grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))

matmul_kernel[grid](
    A, B, C,
    M, N, K,
    BLOCK=BLOCK,
)

print(torch.allclose(C, A @ B, atol=1e-2))