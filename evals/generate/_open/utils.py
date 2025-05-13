

import torch


def reset_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()