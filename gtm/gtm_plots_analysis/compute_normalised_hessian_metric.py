import torch

def pairwise_blockwise_nuclear_normalize_vectorised(hessians, eps=1e-12):
    """Vectorized normalization of Hessians using pairwise 2x2 nuclear norms (off-diagonal only)."""
    # hessians: (N, D, D)
    N, D, _ = hessians.shape
    normed = hessians.clone()

    # Create index grid for i, j
    idx = torch.arange(D)
    ii, jj = torch.meshgrid(idx, idx, indexing='ij')  # shape (D, D)

    # (N, D, D) index each required entry for all blocks
    block00 = hessians[:, ii, ii]   # (N, D, D): hessians[:, i, i] for all i, j
    block01 = hessians[:, ii, jj]   # (N, D, D): hessians[:, i, j]
    block10 = hessians[:, jj, ii]   # (N, D, D): hessians[:, j, i]
    block11 = hessians[:, jj, jj]   # (N, D, D): hessians[:, j, j]

    # Stack: (N, D, D, 2, 2)
    blocks = torch.stack([
        torch.stack([block00, block01], dim=-1),
        torch.stack([block10, block11], dim=-1)
    ], dim=-2)  # (N, D, D, 2, 2)

    # Compute nuclear norm along the last two dims -> (N, D, D)
    block_nuc_norm = torch.linalg.norm(blocks, ord='nuc', dim=(-2, -1)) + eps  # (N,D,D)

    # Off-diagonal mask (D,D)
    offdiag_mask = ~torch.eye(D, dtype=torch.bool, device=hessians.device)
    # (N, D, D): broadcast mask along batch
    mask = offdiag_mask.unsqueeze(0)  # (1, D, D)

    # Only normalize off-diagonals
    normed[:, mask[0]] = hessians[:, mask[0]] / block_nuc_norm[:, mask[0]]

    return normed

#### test compare to:

#
#import torch
#
#def pairwise_blockwise_nuclear_normalize(hessians, eps=1e-12):
#    """
#    For a batch of Hessians (N, D, D), return a matrix (N, D, D) where
#    each (n, i, j) element is hessians[n, i, j] divided by the nuclear norm of
#    the 2x2 block [[hessians[n,i,i], hessians[n,i,j]],
#                   [hessians[n,j,i], hessians[n,j,j]]]
#
#    Off-diagonals normalized, diagonals left unchanged.
#    """
#    N, D, _ = hessians.shape
#    normed = hessians.clone()  # preserve diagonals, will overwrite off-diagonals
#    for i in range(D):
#        for j in range(D):
#            if i == j:
#                continue  # optionally, don't normalize diagonals
#            # shape (N,2,2), batch of 2x2 blocks per sample
#            block = torch.stack([
#                torch.stack([hessians[:, i, i], hessians[:, i, j]], dim=-1),
#                torch.stack([hessians[:, j, i], hessians[:, j, j]], dim=-1)
#            ], dim=-2)
#            block_nuc_norm = torch.linalg.norm(block, ord='nuc', dim=(-2, -1)) + eps  # shape (N,)
#            normed[:, i, j] = hessians[:, i, j] / block_nuc_norm
#    return normed
#
## Example usage:
## hessians = torch.randn(5, 4, 4)   # (N=5, D=4)
## result = pairwise_blockwise_nuclear_normalize(hessians)