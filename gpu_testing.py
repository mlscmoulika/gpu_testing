import torch
import time
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon GPU"
else:
    device = torch.device("cpu")
    device_name = 'CPU'
    
print(device, device_name)

# matrix multiplication test
size = 10000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)
torch.cuda.synchronize()
start_time = time.time()
c1 = torch.matmul(a, b)
torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken for torch.matmul: {end_time - start_time:.6f} seconds")
torch.cuda.synchronize()
start_time = time.time()
c2 = a@b
torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken for @ operator: {end_time - start_time:.6f} seconds")
print("Inference:  The @ and torch.matmul are equivalent and the @ operator is just a shorthand for torch.matmul. The performance should be similar for both operations.")
# more optimizations:
# mm for 2 d matrices.
torch.cuda.synchronize()
start_time = time.time()
c3 = torch.mm(a, b)
torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken for torch.mm: {end_time - start_time:.6f} seconds")
torch.set_float32_matmul_precision('high')  # This is to speed up the matrix multiplication on GPU. It is available in PyTorch 2.0 and later.
torch.cuda.synchronize()
start_time = time.time()
c4 = torch.matmul(a, b)
torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken for torch.matmul with high precision: {end_time - start_time:.6f} seconds")
