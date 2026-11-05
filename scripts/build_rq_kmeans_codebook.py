"""Build standalone RQ-KMeans codebooks from a [N, D] .npy file.

Example:
    python scripts/build_rq_kmeans_codebook.py \
        --input artifacts/content/movie_embeddings.npy \
        --output artifacts/rq_kmeans/codebooks.npz \
        --codebook-sizes 64 64 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

# Allow running this script from a source checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mini_forge.rq_kmeans import fit_rq_kmeans  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build residual K-Means codebooks and SIDs")
    parser.add_argument("--input", type=Path, required=True, help="Input .npy matrix [N, D]")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--codebook-sizes", nargs="+", type=int, default=[64, 64, 64])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    array = np.load(args.input)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        parser.error("input must be a numeric .npy matrix with shape [N, D]")
    vectors = torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)
    result = fit_rq_kmeans(
        vectors,
        args.codebook_sizes,
        iterations=args.iterations,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"codebook_{level}": centers.cpu().numpy() for level, centers in enumerate(result.codebooks)}
    payload["codes"] = result.codes.cpu().numpy()
    np.savez_compressed(args.output, **payload)
    print(f"Saved {len(result.codebooks)} codebooks and {len(result.codes)} SIDs to {args.output}")


if __name__ == "__main__":
    main()
