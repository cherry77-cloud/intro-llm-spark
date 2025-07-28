import torch


# ---------------------- Masks ----------------------
def nopeak_mask(size, device):
    mask = torch.triu(torch.ones((1, size, size), device=device), diagonal=1).bool()
    return ~mask


def create_masks(src, trg, src_pad, trg_pad):
    device = src.device
    src_mask = (src != src_pad).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size, device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask
