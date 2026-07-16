'''
https://leetgpu.com/challenges/vector-addition
solving using two approaches:
- using jax
- using triton
'''
# Creating vector using the following contraints:
'''

    Input vectors A and B have identical lengths
    1 ≤ N ≤ 100,000,000
    Performance is measured with N = 25,000,000

'''
import torch
A = torch.randn(25000, device="cuda")
B = torch.randn(25000, device="cuda")

# JAX implementation
import jax.numpy as jnp
import jax
from jax import dlpack as jax_dlpack
A_jax = jax_dlpack.from_dlpack(A)
B_jax = jax_dlpack.from_dlpack(B)
C = jnp.add(A_jax, B_jax)

# Triton implementation
import triton
import triton.language as tl

@triton.jit
def add(A, B, C, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    chunk = pid * BLOCK + tl.arange(0, BLOCK)
    mask = chunk < N
    a = tl.load(A + chunk, mask=mask, other=0.0)
    b = tl.load(B + chunk, mask=mask, other=0.0)
    c = a + b
    tl.store(C + chunk, c, mask=mask)

N = 25000
C_triton = torch.empty(N, device="cuda")
BLOCK = 1024
grid = (triton.cdiv(N, BLOCK),)
add[grid](A, B, C_triton, N, BLOCK=BLOCK)

C_triton_jax = jax_dlpack.from_dlpack(C_triton)
print(jnp.allclose(C_triton_jax, C, atol=1e-2))