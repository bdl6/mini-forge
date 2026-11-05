"""Optimized Product Quantization (OPQ) for item Semantic IDs.

This is an orthogonal-rotation + product-K-Means implementation. Unlike RQ,
each codebook quantizes a disjoint slice of the rotated vector, not a residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OPQResult:
    rotation: torch.Tensor  # [D, D], row vectors are transformed with X @ rotation
    codebooks: list[torch.Tensor]  # M tables of shape [K, D/M]
    codes: torch.Tensor  # [N, M]
    reconstruction: torch.Tensor  # [N, D], back in the input space


@torch.no_grad()
def encode_opq(
    vectors: torch.Tensor,
    rotation: torch.Tensor,
    codebooks: list[torch.Tensor],
    *,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign codes and reconstruct vectors with fitted OPQ parameters."""
    if vectors.ndim != 2 or vectors.shape[0] == 0 or not vectors.is_floating_point():
        raise ValueError("vectors must be nonempty floating-point [N, D]")
    if batch_size < 1 or not codebooks:
        raise ValueError("batch_size and codebooks must be positive/nonempty")
    n, dim = vectors.shape
    if dim % len(codebooks) != 0 or rotation.shape != (dim, dim):
        raise ValueError("rotation and number of codebooks must match input dimension")
    subdim = dim // len(codebooks)
    if any(table.ndim != 2 or table.shape[1] != subdim or table.shape[0] < 1 for table in codebooks):
        raise ValueError("each codebook must have shape [K, D/M]")
    if rotation.device != vectors.device or any(table.device != vectors.device for table in codebooks):
        raise ValueError("vectors, rotation, and codebooks must be on the same device")

    rotated = vectors @ rotation
    codes = []
    reconstructed_parts = []
    for level, table in enumerate(codebooks):
        part = rotated[:, level * subdim:(level + 1) * subdim]
        center_norm = table.square().sum(dim=1).unsqueeze(0)
        assignments = []
        for batch in part.split(batch_size):
            distances = batch.square().sum(dim=1, keepdim=True) + center_norm - 2 * batch @ table.T
            assignments.append(distances.argmin(dim=1))
        code = torch.cat(assignments)
        codes.append(code)
        reconstructed_parts.append(table[code])
    reconstruction = torch.cat(reconstructed_parts, dim=1) @ rotation.T
    return torch.stack(codes, dim=1), reconstruction


@torch.no_grad()
def _fit_kmeans(
    values: torch.Tensor,
    clusters: int,
    iterations: int,
    batch_size: int,
    rng: torch.Generator,
) -> torch.Tensor:
    n, dim = values.shape
    if n >= clusters:
        indices = torch.randperm(n, generator=rng)[:clusters]
    else:
        indices = torch.randint(n, (clusters,), generator=rng)
    centers = values[indices.to(values.device)].clone()
    for _ in range(iterations):
        counts = torch.zeros(clusters, device=values.device, dtype=torch.long)
        sums = torch.zeros(clusters, dim, device=values.device, dtype=values.dtype)
        center_norm = centers.square().sum(dim=1).unsqueeze(0)
        for batch in values.split(batch_size):
            distances = batch.square().sum(dim=1, keepdim=True) + center_norm - 2 * batch @ centers.T
            assignments = distances.argmin(dim=1)
            counts += torch.bincount(assignments, minlength=clusters)
            sums.index_add_(0, assignments, batch)
        occupied = counts > 0
        centers[occupied] = sums[occupied] / counts[occupied].unsqueeze(1)
        # Unused centers retain their prior values; no random movement of empty codes.
    return centers


@torch.no_grad()
def fit_opq(
    vectors: torch.Tensor,
    num_codebooks: int,
    codebook_size: int,
    *,
    rotation_iterations: int = 10,
    kmeans_iterations: int = 30,
    batch_size: int = 4096,
    seed: int = 0,
) -> OPQResult:
    """Alternate product K-Means and orthogonal Procrustes rotation updates.

    With Z = X @ R and PQ reconstruction Q, the rotation update solves
    min_R ||X @ R - Q||_F, subject to R.T @ R = I. If X.T @ Q = U S V.T,
    the optimum is R = U @ V.T. A final PQ fit matches the final rotation.
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0 or not vectors.is_floating_point():
        raise ValueError("vectors must be nonempty floating-point [N, D]")
    if not torch.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or infinity")
    n, dim = vectors.shape
    if num_codebooks < 1 or dim % num_codebooks != 0:
        raise ValueError("num_codebooks must divide input dimension")
    if codebook_size < 1 or rotation_iterations < 1 or kmeans_iterations < 1 or batch_size < 1:
        raise ValueError("sizes and iteration counts must be positive")

    rng = torch.Generator(device="cpu").manual_seed(seed)
    rotation = torch.eye(dim, device=vectors.device, dtype=vectors.dtype)
    subdim = dim // num_codebooks

    def fit_tables(current_rotation: torch.Tensor) -> list[torch.Tensor]:
        transformed = vectors @ current_rotation
        return [
            _fit_kmeans(
                transformed[:, level * subdim:(level + 1) * subdim],
                codebook_size, kmeans_iterations, batch_size, rng,
            )
            for level in range(num_codebooks)
        ]

    for _ in range(rotation_iterations):
        tables = fit_tables(rotation)
        codes, _ = encode_opq(vectors, rotation, tables, batch_size=batch_size)
        quantized_rotated = torch.cat([table[codes[:, level]] for level, table in enumerate(tables)], dim=1)
        # Float64 SVD avoids orthogonality drift with a 384x384 CUDA rotation.
        covariance = (vectors.T @ quantized_rotated).to(torch.float64)
        left, _, right_t = torch.linalg.svd(covariance, full_matrices=False)
        rotation = (left @ right_t).to(vectors.dtype)

    tables = fit_tables(rotation)
    codes, reconstruction = encode_opq(vectors, rotation, tables, batch_size=batch_size)
    return OPQResult(rotation=rotation, codebooks=tables, codes=codes, reconstruction=reconstruction)
