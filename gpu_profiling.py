# summary of this file: 
'''
inpired from : https://www.youtube.com/watch?v=LuhJEEJQgUM
to capture all I know so far about gpu profiling. 
'''
# Method 1: Using torch.profiler
import torch
from torch.profiler import profile, record_function, ProfilerActivity
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        c = a+b
print("Method 1 - torch.profiler:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Method 2: Using torch.cuda.Event
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize() 
start.record()
c=a+b
end.record()
torch.cuda.synchronize()
print("Method 2 - torch.cuda.Event:")
print(f"Time taken for addition: {start.elapsed_time(end)} milliseconds")

# advatage of using torch.profiler is that it gives you a detailed breakdown of the time taken by each operation, 
#while torch.cuda.Event only gives you the total time taken for a specific operation. 
#However, torch.profiler can introduce some overhead, so for very small operations, the timing may not be accurate.

# Method 3: torch.compile
def vector_addition(a, b):
    return a + b
compiled_vector_addition = torch.compile(vector_addition)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
c = compiled_vector_addition(a, b)
end.record()
print("Method 3 - torch.compile:")
print(f"Time taken for compiled addition: {start.elapsed_time(end)} milliseconds")

