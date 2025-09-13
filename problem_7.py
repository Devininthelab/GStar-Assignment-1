import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    q_block = q_block.to(tl.float32)  # Convert to float32 for computation
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    window_start = max(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
    # You should have 3 phases:
    # 1. Phase 0: Sink blocks that are before the sliding window
    for start_n in range(0, window_start, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED (Part 3: SWA Logic)
        # Hint: You might need to apply the per-element sliding window mask to s_ij.
        #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = k_block.to(tl.float32)

         # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0) # shape [BLOCK_N, HEAD_DIM]
        v_block = v_block.to(tl.float32)  # Convert to float32 for computation

        #  2. Compute the attention scores (S_ij).
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block) * qk_scale

        # Apply sliding window mask
        col = k_offsets[None, :]
        row = q_offsets[:, None]
        sliding = (col <= row) & (col >= row - (WINDOW_SIZE - 1))
        # 2) sink at start:   j < sink_size  *and*  j <= i
        sink = (col < SINK_SIZE) & (col <= row)
        s_ij = tl.where(sliding | sink, s_ij, -1e9)

        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
        # Implement the online softmax update logic.
        # 1. Find the new running maximum (`m_new`).
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))

        # 2. Rescale the existing accumulator (`acc`) and denominator (`l_i`).
        l_i_normalized_factor = tl.exp2(m_i - m_new)
        acc_normalized = acc * l_i_normalized_factor[:, None]
        
        # 3. Compute the attention probabilities for the current tile (`p_ij`).
        p_ij = tl.exp2(s_ij - m_new[:, None]) # equals P_tilde_ij in the old code

        # 4. Update the accumulator `acc` using `p_ij` and `v_block`.
        acc = acc_normalized + tl.dot(p_ij, v_block)

        # 5. Update the denominator `l_i`.
        l_i = l_i * l_i_normalized_factor + tl.sum(p_ij, axis=1)
        # 6. Update the running maximum `m_i` for the next iteration.
        m_i = m_new

    # 2. Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED (Part 3: SWA Logic)
        # Hint: You might need to apply the per-element sliding window mask to s_ij.
        #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = k_block.to(tl.float32)

         # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0) # shape [BLOCK_N, HEAD_DIM]
        v_block = v_block.to(tl.float32)  # Convert to float32 for computation

        #  2. Compute the attention scores (S_ij).
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block) * qk_scale

        # Apply sliding window mask
        col = k_offsets[None, :]
        row = q_offsets[:, None]
        sliding = (col <= row) & (col >= row - (WINDOW_SIZE - 1))
        # 2) sink at start:   j < sink_size  *and*  j <= i
        sink = (col < SINK_SIZE) & (col <= row)
        s_ij = tl.where(sliding | sink, s_ij, -1e9)

        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
        # Implement the online softmax update logic.
        # 1. Find the new running maximum (`m_new`).
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))

        # 2. Rescale the existing accumulator (`acc`) and denominator (`l_i`).
        l_i_normalized_factor = tl.exp2(m_i - m_new)
        acc_normalized = acc * l_i_normalized_factor[:, None]
        
        # 3. Compute the attention probabilities for the current tile (`p_ij`).
        p_ij = tl.exp2(s_ij - m_new[:, None]) # equals P_tilde_ij in the old code

        # 4. Update the accumulator `acc` using `p_ij` and `v_block`.
        acc = acc_normalized + tl.dot(p_ij, v_block)

        # 5. Update the denominator `l_i`.
        l_i = l_i * l_i_normalized_factor + tl.sum(p_ij, axis=1)
        # 6. Update the running maximum `m_i` for the next iteration.
        m_i = m_new

    # 3. Phase 2: Diagonal Blocks
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # STUDENT IMPLEMENTATION REQUIRED
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = k_block.to(tl.float32)

        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0) # shape [BLOCK_N, HEAD_DIM]
        v_block = v_block.to(tl.float32)  # Convert to float32 for computation

        #  2. Compute the attention scores (S_ij).
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block) * qk_scale
        
        # Apply sliding window mask
        col = k_offsets[None, :]
        row = q_offsets[:, None]
        sliding = (col <= row) & (col >= row - (WINDOW_SIZE - 1))
        # 2) sink at start:   j < sink_size  *and*  j <= i
        sink = (col < SINK_SIZE) & (col <= row)
        s_ij = tl.where(sliding | sink, s_ij, -1e9)


        # 3. Update the online softmax statistics (m_i, l_i) and the accumulator (acc).
        # Implement the online softmax update logic.
        # 1. Find the new running maximum (`m_new`).
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))

        # 2. Rescale the existing accumulator (`acc`) and denominator (`l_i`).
        l_i_normalized_factor = tl.exp2(m_i - m_new)
        acc_normalized = acc * l_i_normalized_factor[:, None]
        
        # 3. Compute the attention probabilities for the current tile (`p_ij`).
        p_ij = tl.exp2(s_ij - m_new[:, None]) # equals P_tilde_ij in the old code

        # 4. Update the accumulator `acc` using `p_ij` and `v_block`.
        acc = acc_normalized + tl.dot(p_ij, v_block)

        # 5. Update the denominator `l_i`.
        l_i = l_i * l_i_normalized_factor + tl.sum(p_ij, axis=1)
        # 6. Update the running maximum `m_i` for the next iteration.
        m_i = m_new
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return o