'''
https://leetgpu.com/challenges/matrix-multiplication
solving using two approaches:
- using jax
- using triton
'''
# Creating matrix using the following contraints:
'''

    1 ≤ M, N, K ≤ 8192
'''
import torch
M, N, K = 1000, 1000, 1000
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")        

# JAX implementation
import jax.numpy as jnp
import jax
from jax import dlpack as jax_dlpack
A_jax = jax_dlpack.from_dlpack(A)
B_jax = jax_dlpack.from_dlpack(B)
C = jnp.matmul(A_jax, B_jax)

# Triton implementation
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
    off_k = tl.arange(0, BLOCK)

    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    for k in range(0, K, BLOCK):
        a = tl.load(A + rows[:, None] * K + (k+off_k)[None, :],
                    mask=(rows[:, None] < M) & ((k+off_k)[None, :]<K),
                    other=0.0)

        b = tl.load(B + (k+off_k)[:, None] * N + cols[None, :],
                    mask=(cols[None, :] < N) & ((k+off_k)[None, :]<K),
                    other=0.0)

        acc += tl.dot(a, b)

    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )

C_triton = torch.empty((M, N), device="cuda")
BLOCK = 64
grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
matmul_kernel[grid](
    A, B, C_triton,
    M, N, K,
    BLOCK=BLOCK,
)
C_triton_jax = jax_dlpack.from_dlpack(C_triton)
print(jnp.allclose(C_triton_jax, C, atol=1e-2))
print(C_triton[:5, :5])
print(C[:5, :5])