"""Standalone residual K-Means codebooks for semantic IDs.

Input vectors are clustered directly. Pass RQ-VAE encoder outputs instead of raw
item embeddings if the codebooks should live in the RQ-VAE latent space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RQKMeansResult:
    # Each codebook has shape [K, D]; sizes may differ between levels.
    codebooks: list[torch.Tensor]
    # codes[n, level] selects a row from codebooks[level].
    codes: torch.Tensor


@torch.no_grad()
def encode_rq_kmeans(
    vectors: torch.Tensor,
    codebooks: list[torch.Tensor],
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign each vector to one code at each residual level.

    Returns codes [N, L] and the reconstructed vectors [N, D].
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("vectors must have shape [N, D] with N > 0")
    if not codebooks or batch_size < 1:
        raise ValueError("codebooks must be nonempty and batch_size must be positive")
    if any(c.ndim != 2 or c.shape[0] == 0 or c.shape[1] != vectors.shape[1] for c in codebooks):
        raise ValueError("each codebook must have shape [K, D] with the input D")
    if any(c.device != vectors.device for c in codebooks):
        raise ValueError("vectors and codebooks must be on the same device")

    residual = vectors.clone()
    reconstruction = torch.zeros_like(vectors)
    level_codes = []
    for centers in codebooks:
        assignments = []
        center_norm = centers.square().sum(dim=1).unsqueeze(0)
        for batch in residual.split(batch_size):
            # ||x-c||² = ||x||² + ||c||² - 2*x@c.T; no [N,K,D] tensor.
            distances = batch.square().sum(dim=1, keepdim=True) + center_norm - 2 * batch @ centers.T
            assignments.append(distances.argmin(dim=1))
        code = torch.cat(assignments)
        selected = centers[code]
        reconstruction += selected
        residual -= selected
        level_codes.append(code)
    return torch.stack(level_codes, dim=1), reconstruction


@torch.no_grad()
def fit_rq_kmeans(
    vectors: torch.Tensor,
    codebook_sizes: list[int],
    *,
    iterations: int = 30,
    batch_size: int = 4096,
    seed: int = 0,
) -> RQKMeansResult:
    """Fit one K-Means codebook to each successive residual.

    vectors: [N, D] float tensor. codebook_sizes: e.g. [64, 64, 64].
    The input is never modified. All outputs stay on the input device.
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0 or not vectors.is_floating_point():
        raise ValueError("vectors must be a nonempty floating-point [N, D] tensor")
    if not torch.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or infinity")
    if not codebook_sizes or any(k < 1 for k in codebook_sizes):
        raise ValueError("codebook_sizes must contain positive integers")
    if iterations < 1 or batch_size < 1:
        raise ValueError("iterations and batch_size must be positive")

    rng = torch.Generator(device="cpu").manual_seed(seed)
    residual = vectors.clone()
    codebooks: list[torch.Tensor] = []
    for k in codebook_sizes:
        n, dim = residual.shape
        # Sample actual vectors so the initial centers are in the data range.
        if n >= k:
            initial_indices = torch.randperm(n, generator=rng)[:k]
        else:
            initial_indices = torch.randint(n, (k,), generator=rng)
        centers = residual[initial_indices.to(residual.device)].clone()

        for _ in range(iterations):
            counts = torch.zeros(k, device=residual.device, dtype=torch.long)
            sums = torch.zeros(k, dim, device=residual.device, dtype=residual.dtype)
            center_norm = centers.square().sum(dim=1).unsqueeze(0)
            for batch in residual.split(batch_size):
                distances = batch.square().sum(dim=1, keepdim=True) + center_norm - 2 * batch @ centers.T
                assignment = distances.argmin(dim=1)
                counts += torch.bincount(assignment, minlength=k)
                sums.index_add_(0, assignment, batch)

            occupied = counts > 0
            centers[occupied] = sums[occupied] / counts[occupied].unsqueeze(1)
            # A cluster with no members retains its previous center.

        # Reassign with the final centers: assignments from the last iteration
        # were computed before its center update.
        assignments = []
        center_norm = centers.square().sum(dim=1).unsqueeze(0)
        for batch in residual.split(batch_size):
            distances = batch.square().sum(dim=1, keepdim=True) + center_norm - 2 * batch @ centers.T
            assignments.append(distances.argmin(dim=1))
        code = torch.cat(assignments)
        codebooks.append(centers)
        residual -= centers[code]

    # Re-encode from the original vectors so codes match the saved codebooks.
    codes, _ = encode_rq_kmeans(vectors, codebooks, batch_size)
    return RQKMeansResult(codebooks=codebooks, codes=codes)
