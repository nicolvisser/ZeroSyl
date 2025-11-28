import torch


def is_contiguous(segments: torch.Tensor):
    """
    Determine whether the segmenation is contiguous (True) or gappy (False).
        segments[:, 0] = start times
        segments[:, 1] = end times
        segments[:, 2] = predicted id
    """
    ends = segments[:-1, 1]
    next_starts = segments[1:, 0]
    return torch.all(ends == next_starts).item()
