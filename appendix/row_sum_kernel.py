import torch
import triton
import triton.language as tl


# Compute this formula: 
# Given a 2D matrix of shape (M, D), compute an output vector of shape M where
# output[i] = sum_{j=0}^{D-1} X[i, j]

@triton.jit
def row_sum_kernel(
    X_ptr,       # Pointer to the input tensor
    Output_ptr,  # Pointer to the output vector
    M, D,       # Dimensions of the input tensor
    BLOCK_SIZE_D: tl.constexpr  # Block size for the kernel
):
    
    # Get the unique row index for this program instance
    pid_m = tl.program_id(axis=0)  # Row index in the grid (0 to M-1). Here we only take the row index as we from 2D to 1D output
    # triton launches a grid of programs, one per row. pid_m is the row index this program will handle.
    '''
    Grid along axis=0
    Row 0  -> program 0
    Row 1  -> program 1
    Row 2  -> program 2
    ...
    Row M-1 -> program M-1
    '''
    # Each program independently sums one row

    # Calculate the starting pointer for the assigned row in X
    row_start_ptr = X_ptr + pid_m * D

    # Initialize an accumulator in SRAM for the sum
    accumulator = 0.0

    # Loop over the row in blocks (titles):
    for d_offset in range(0, tl.cdiv(D, BLOCK_SIZE_D)): # tl.cdiv(D, BLOCK_SIZE_D) retiurns the ceiling of D / BLOCK_SIZE_D
        # Create offsets and a mask for the current tile.
        d_offsets = d_offset * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D) # length from 0 to block size D, for example BLOCK_SIZE_D = 1024, d_offsets = [0, 1, 2, ..., 1023] if d_offset = 0
        d_mask = d_offsets < D


        # Load a tile of the X row from HBM to SRAM
        x_tile = tl.load(row_start_ptr + d_offsets, mask=d_mask, other=0.0)

        # Perform computation on the tile in SRAM
        accumulator += tl.sum(x_tile)

    # Store the final accumulated value from SRAM back to HBM
    tl.store(Output_ptr + pid_m, accumulator)


def row_sum_triton(X):
    """
    Python launcher function for the Triton kernel.
    """
    M, D = X.shape
    # Allocate output tensor
    output = torch.empty(M, device=X.device, dtype=X.dtype)

    grid = (M,) # One program per row
    row_sum_kernel[grid](
        X_ptr=X,
        Output_ptr=output,
        M=M,
        D=D,
        BLOCK_SIZE_D=1024,  # You can tune this parameter
    )
    return output


def row_sum_pytorch(X):
    """
    Calculates the sum of each row using a sequential Python loop. (similar to
        torch.sum(X, dim=1)
    This is intentionally inefficient on a GPU to contrast with the Triton kernel.
    """
    M = X.shape[0]
    output = torch.empty(M, device=X.device, dtype=X.dtype)
    # This `for` loop runs sequentially, one row at a time.
    for i in range(M):
        output[i] = torch.sum(X[i])
    return output


# --- Verification Script ---
X_test = torch.randn(256, 3000, device='cuda', dtype=torch.float32)
triton_result = row_sum_triton(X_test)
pytorch_result = row_sum_pytorch(X_test)

# Use appropriate tolerance for floating-point comparison
print(f"Results are close: {torch.allclose(triton_result, pytorch_result, rtol=1e-4, atol=1e-4)}")
