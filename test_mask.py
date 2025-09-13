import torch


def create_mask_bool(
    seq_len: int,
    window_size: int,
    sink_size: int,
    device=None
    ) -> torch.Tensor:
    
    idx = torch.arange(seq_len, device=device)
    row = idx.unsqueeze(1)   # (seq_len, 1)
    col = idx.unsqueeze(0)   # (1, seq_len)

    # 1) sliding window:  i - (window_size-1) <= j <= i
    sliding = (col <= row) & (col >= row - (window_size - 1))

    # 2) sink at start:   j < sink_size  *and*  j <= i
    sink = (col < sink_size) & (col <= row)

    return sliding | sink


print(create_mask_bool(8, 2, 0))