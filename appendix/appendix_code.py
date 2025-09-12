import triton
import triton.language as tl
import torch

@triton.jit
def get_program_ids_kernel(output_ptr, M, N):
    # Get the unique 2D coordinate for this program instance
    pid_m = tl.program_id(axis=0) # The row index in the grid (0 to M-1)
    pid_n = tl.program_id(axis=1) # The column index in the grid (0 to N-1)

    # Calculate the linear memory offset for this 2D coordinate
    offset = pid_m * N + pid_n

    # Store a unique value based on the ID, e.g., 1002 for (1, 2)
    unique_id = pid_m * 1000 + pid_n
    tl.store(output_ptr + offset, unique_id)




# --- Verification ---

M, N = 4, 5
output = torch.empty(M * N, device='cuda', dtype=torch.int32)

# Launch the kernel on a 2D grid of size (M, N)
get_program_ids_kernel[(M, N)](output, M, N)

# Reshape the output to see the 2D grid of unique IDs
print(output.reshape(M, N))
# Expected output:
# tensor([[ 0, 1, 2, 3, 4],
# [1000, 1001, 1002, 1003, 1004],
# [2000, 2001, 2002, 2003, 2004],
# [3000, 3001, 3002, 3003, 3004]], device='cuda:0', dtype=torch.int32)


'''
Spwan mxn programs in a 2D grid
- Each program has a unique 2D coordinate (pid_m, pid_n)
- Imagine as a grid of threads (programs), then each thread run a job. Each thread also has a unique ID, which can be accessed by offset = pid_m * N + pid_n
- So the unique_id is for compute the formula, and then the tl.store will store the value 
'''