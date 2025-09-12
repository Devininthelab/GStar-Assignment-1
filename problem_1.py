import torch
import torch.nn as nn
import math

class FlashAttention2Function(torch.autograd.Function):
    """
    A pure PyTorch implementation of the FlashAttention-2 forward pass.
    This version is a template for student implementation.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get dimensions from input tensors following the (B, H, N, D) convention
        B, H, N_Q, D_H = Q.shape   # (Batch, Heads, Query length, Head dim)
        _, _, N_K, _ = K.shape  # (Batch, Heads, Key length, Head dim)

        # Define tile sizes; tile = block
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 128
        
        N_Q_tiles = math.ceil(N_Q / Q_TILE_SIZE)    # Number of blocks 
        N_K_tiles = math.ceil(N_K / K_TILE_SIZE)

        # Initialize final output tensors
        O_final = torch.zeros_like(Q, dtype=Q.dtype)
        L_final = torch.zeros((B, H, N_Q), device=Q.device, dtype=torch.float32) # Normalization factors
        
        scale = 1.0 / math.sqrt(D_H)

        # Main loops: Iterate over each batch and head
        for b in range(B): # For each batch
            for h in range(H): # For each head
                Q_bh = Q[b, h, :, :]
                K_bh = K[b, h, :, :]
                V_bh = V[b, h, :, :]

                # Loop over query tiles: main code
                for i in range(N_Q_tiles):
                    q_start = i * Q_TILE_SIZE
                    q_end = min((i + 1) * Q_TILE_SIZE, N_Q) # Plus 1 due to Python indexing
                    Q_tile = Q_bh[q_start:q_end, :]

                    # Initialize accumulators for this query tile
                    o_i = torch.zeros_like(Q_tile, dtype=Q.dtype)    
                    l_i = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32) # a vec of size Q_TILE_SIZE
                    m_i = torch.full((q_end - q_start,), -float('inf'), device=Q.device, dtype=torch.float32) # running_max so far

                    # Inner loop over key/value tiles
                    for j in range(N_K_tiles): # number of blocks
                        k_start = j * K_TILE_SIZE
                        k_end = min((j + 1) * K_TILE_SIZE, N_K)

                        K_tile = K_bh[k_start:k_end, :]
                        V_tile = V_bh[k_start:k_end, :]
                        
                        S_ij = (Q_tile @ K_tile.transpose(-1, -2)) * scale # a block's attention scores
                        
                        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
                        # 1. Apply causal masking if is_causal is True.
                        
                        if is_causal: 
                            mask = torch.triu(torch.ones((q_end - q_start, k_end - k_start), device=Q.device), diagonal=1 + (i * Q_TILE_SIZE) - (j * K_TILE_SIZE)).bool()
                            S_ij = S_ij.masked_fill(mask, float('-inf'))

                        # 2. Compute the new running maximum: for more visualization, see : https://github.com/hkproj/triton-flash-attention/blob/main/notes/0004%20-%20Block%20Matrix%20Multiplication.pdf
                        m_new = torch.maximum(m_i, torch.amax(S_ij, dim=-1))
            
                        # 3. Rescale the previous accumulators (o_i, l_i)
                        l_i_normalized_factor = torch.exp(m_i - m_new) 

                        # 4. Compute the probabilities for the current tile, P_tilde_ij = exp(S_ij - m_new).
                        P_tilde_ij = torch.exp(S_ij - m_new.unsqueeze(-1)).to(Q.dtype)

                        # 5. Accumulate the current tile's contribution to the accumulators to update l_i and o_i
                        l_i = l_i_normalized_factor * l_i + torch.sum(P_tilde_ij, dim=-1).float()
                        o_i = torch.diag(l_i_normalized_factor).to(Q.dtype) @ o_i + P_tilde_ij @ V_tile

                        # 6. Update the running max for the next iteration
                        m_i = m_new
                        
                        # --- END OF STUDENT IMPLEMENTATION ---

                    # After iterating through all key tiles, normalize the output
                    # This part is provided for you. It handles the final division safely.
                    l_i_reciprocal = torch.where(l_i > 0, 1.0 / l_i, 0)
                    o_i_normalized = o_i * l_i_reciprocal.unsqueeze(-1)
                    
                    L_tile = m_i + torch.log(l_i)
                    
                    # Write results for this tile back to the final output tensors
                    O_final[b, h, q_start:q_end, :] = o_i_normalized
                    L_final[b, h, q_start:q_end] = L_tile
        
        O_final = O_final.to(Q.dtype)

        ctx.save_for_backward(Q, K, V, O_final, L_final)
        ctx.is_causal = is_causal
 
        return O_final, L_final
    
    @staticmethod
    def backward(ctx, grad_out, grad_L):
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")
    

if __name__ == "__main__":
    print(torch.zeros(10, device='cuda'))
    print(torch.full((3,), -float('inf'), dtype=torch.float32))
    print(torch.triu(torch.ones(4, 4), diagonal=1))
    print(torch.full((4,), -float('inf'), device='cuda', dtype=torch.float32).unsqueeze(-1))

    '''
    Exaplaination for the causal mask:
    Since we are computing attention in tiles(blocks):
        - Q queries is split into tiles of size Q_TILE_SIZE
        - K keys is split into tiles of size K_TILE_SIZE

    Then when we loop over the tiles: i for Q tiles, j for K tiles -> compute scores, then need mask


    We have:
    i -> index of the current Q tile
    j -> index of the current K tile
    then the row and column indices in the full attention matrix are: 
        - row indices in full matrix: q_start = i * Q_TILE_SIZE -> q_end = (i + 1) * Q_TILE_SIZE
        - col indices in full matrix: k_start = j * K_TILE_SIZE -> k_end = (j + 1) * K_TILE_SIZE
    
    Note that:
        - Q_tile has shape (Q_TILE_SIZE = q_end - q_start, D_H)
        - K_tile has shape (K_TILE_SIZE = k_end - k_start, D_H)
        => we need a mask for this title, which has shape (Q_TILE_SIZE, K_TILE_SIZE)

    In a full matrix, the masking rule is: 
    mask[q, k] = True if k > q else False for q, k is the FULL MATRIX indices

    Query index in the full sequence:
        - q = i * Q_TILE_SIZE + r (q_offset)
    Key index in the full sequence:
        - k = j * K_TILE_SIZE + c (k_offset)

    So the masking rule becomes:
        - mask[r, c] = True if (j * K_TILE_SIZE + c) > (i * Q_TILE_SIZE + r) else False for r, c in the TILE indices

    Rerranging for local masks:
        - c > r + (i * Q_TILE_SIZE) - (j * K_TILE_SIZE); which is equivalent to: c > r + d 
    
    Then based on torch.triu documentation:
        - torch.triu(input, diagonal=d): mask all entries where c >= r + d
        - So we need to set diagonal = 1 + (i * Q_TILE_SIZE) - (j * K_TILE_SIZE) to get the correct mask

    THIS HAS EXPLAINED LINE 63
        
    '''