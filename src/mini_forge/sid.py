from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .rqvae import load_rqvae
from .rq_kmeans import fit_rq_kmeans, encode_rq_kmeans
from .opq import fit_opq
from .utils import ensure_dir, resolve_device, seed_everything, write_json


def _codebook_stats(codes: np.ndarray, codebook_size: int) -> list[dict[str, float]]:
    """统计每级码本的使用率、未使用码比例及使用分布的困惑度。

    codes 形状为 [电影数, 量化级数]，每个元素是该级选中的码编号。
    """
    result: list[dict[str, float]] = []
    for level in range(codes.shape[1]):
        # bincount 统计本级每个码被多少部电影选中；minlength 保留未使用的码。
        counts = np.bincount(codes[:, level], minlength=codebook_size).astype(np.float64)
        # 只对用过的码计算概率，避免对 0 取 log。
        probabilities = counts[counts > 0] / counts.sum()
        used = int((counts > 0).sum())
        result.append({
            "level": level + 1,
            "utilization": used / codebook_size,
            "dead_code_rate": 1.0 - used / codebook_size,
            # exp(信息熵) 是“有效使用码数”；分布越均匀，值越高。
            "perplexity": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
        })
    return result


def _genre_jaccard(movies: pd.DataFrame, codes: np.ndarray, seed: int) -> dict[str, float]:
    """检查共享 SID 前缀的电影是否更可能拥有相同类型。"""
    # 将 "Action|Comedy" 拆成集合，便于计算两部电影的类型交并比。
    genres = [set(str(value).split("|")) for value in movies.genres]
    rng = np.random.default_rng(seed)

    def average_for_prefix(levels: int, cap: int = 20000) -> float:
        """对共享前 levels 级码的电影对，计算平均 Jaccard 相似度。"""
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index, code in enumerate(codes):
            # 前缀相同的电影落在同一个桶中；桶里保存电影的行号。
            buckets[tuple(int(x) for x in code[:levels])].append(index)
        pairs: list[tuple[int, int]] = []
        for indices in buckets.values():
            # combinations(indices, 2) 枚举桶内所有不重复的两电影组合。
            pairs.extend(itertools.combinations(indices, 2))
            if len(pairs) >= cap:
                break
        if not pairs:
            return 0.0
        if len(pairs) > cap:
            chosen = rng.choice(len(pairs), cap, replace=False)
            pairs = [pairs[index] for index in chosen]
        # Jaccard = 共同类型数 / 合并后的类型数；值越高表示类型越相近。
        return float(np.mean([len(genres[a] & genres[b]) / max(1, len(genres[a] | genres[b])) for a, b in pairs]))

    # 随机电影对提供对照值，用于观察“共享前缀”是否优于随机配对。
    random_pairs = rng.integers(0, len(genres), size=(min(20000, len(genres) * 5), 2))
    random_score = float(np.mean([len(genres[a] & genres[b]) / max(1, len(genres[a] | genres[b])) for a, b in random_pairs if a != b]))
    return {"same_l1": average_for_prefix(1), "same_l1_l2": average_for_prefix(min(2, codes.shape[1])), "random": random_score}


def _neighbor_recall(original: np.ndarray, quantized: np.ndarray, k: int = 10) -> float:
    """比较原内容空间与量化潜空间的最近邻，衡量邻居关系保留程度。"""
    # 两组向量可以有不同维度；各自按行归一化后计算本空间的余弦相似度。
    original = original / np.maximum(np.linalg.norm(original, axis=1, keepdims=True), 1e-12)
    quantized = quantized / np.maximum(np.linalg.norm(quantized, axis=1, keepdims=True), 1e-12)
    # 矩阵乘转置得到所有电影两两之间的相似度表 [N, N]。
    original_similarity = original @ original.T
    quantized_similarity = quantized @ quantized.T
    # 对角线是电影与自身的相似度；排除自己，避免占用近邻名额。
    np.fill_diagonal(original_similarity, -np.inf)
    np.fill_diagonal(quantized_similarity, -np.inf)
    # argpartition 只找最大的 k 个位置，不要求这 k 个位置内部排好序。
    original_top = np.argpartition(original_similarity, -k, axis=1)[:, -k:]
    quantized_top = np.argpartition(quantized_similarity, -k, axis=1)[:, -k:]
    # 对每部电影算两组近邻的交集比例，再对全部电影取平均。
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(original_top, quantized_top)]))


def export_rqvae_sid(config: dict[str, Any]) -> dict[str, Any]:
    """用训练好的 RQ-VAE 为电影导出唯一 SID，并生成质量报告。"""
    device = resolve_device(config.get("device", "auto"))
    artifact_root = Path(config["paths"]["artifact_dir"])
    output_dir = ensure_dir(artifact_root / "sid")
    # embed 产物按相同行顺序保存：embeddings[i] 对应 movie_ids[i]。
    embeddings = np.load(artifact_root / "content" / "movie_embeddings.npy").astype(np.float32)
    movie_ids = np.load(artifact_root / "content" / "movie_ids.npy").astype(np.int64)
    # 按 movie_ids 顺序重排电影表，使类型指标与 codes 的行号一一对应。
    movies = pd.read_csv(Path(config["paths"]["processed_dir"]) / "movies.csv").set_index("movie_id").loc[movie_ids].reset_index()
    model = load_rqvae(artifact_root / "rqvae" / "best.pt", device)
    # 推断不需要梯度；model.eval() 已在 load_rqvae 中完成，也不会更新 EMA。
    with torch.inference_mode():
        output = model(torch.from_numpy(embeddings).to(device))
    # 转回 CPU/NumPy，后续计算质量指标并写入 .npy、JSON 文件。
    codes = output["codes"].cpu().numpy().astype(np.int64)
    reconstructed = output["reconstruction"].cpu().numpy()
    quantized = output["quantized"].cpu().numpy()

    # 不同电影可能得到完全相同的 L 级码；先按码组合分桶。
    collision_indices = np.zeros(len(movie_ids), dtype=np.int64)
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, code in enumerate(codes):
        buckets[tuple(int(value) for value in code)].append(index)
    # 同桶电影按 movie_id 排序，依次赋 C_0、C_1、...，使编号稳定且 SID 唯一。
    for indices in buckets.values():
        for collision, index in enumerate(sorted(indices, key=lambda value: int(movie_ids[value]))):
            collision_indices[index] = collision
    mapping: dict[str, Any] = {}
    for movie_id, code, collision in zip(movie_ids, codes, collision_indices):
        # 例如 [4, 42, 57] + 碰撞号 0 -> <L1_4> <L2_42> <L3_57> <C_0>。
        tokens = [f"<L{level + 1}_{int(value)}>" for level, value in enumerate(code)] + [f"<C_{int(collision)}>" ]
        mapping[str(int(movie_id))] = {"codes": [int(value) for value in code], "collision": int(collision), "tokens": tokens}
    # movie_to_sid.json 是生成器训练和推荐结果回表时使用的核心映射。
    write_json(output_dir / "movie_to_sid.json", mapping)
    np.save(output_dir / "codes.npy", codes)
    np.save(output_dir / "quantized_embeddings.npy", quantized.astype(np.float32))
    # 碰撞率统计的是“落在多电影桶中的电影比例”，不是额外碰撞电影数比例。
    collision_movies = sum(len(indices) for indices in buckets.values() if len(indices) > 1)
    # 将 prepare 的共现电影对与当前 SID 对齐，用于协同质量指标。
    raw_pairs = np.load(Path(config["paths"]["processed_dir"]) / "i2i_pairs.npy")
    code_by_movie = {int(movie_id): code for movie_id, code in zip(movie_ids, codes)}
    collaborative_rows = [
        (code_by_movie[int(a)], code_by_movie[int(b)], float(weight))
        for a, b, weight in raw_pairs if int(a) in code_by_movie and int(b) in code_by_movie
    ]
    if collaborative_rows:
        # 共现权重越大，通常代表两电影在用户训练序列中更常靠近出现。
        pair_weights = np.asarray([row[2] for row in collaborative_rows])
        # 分别统计共现对共享第一级码、共享前两级码的比例。
        shared_l1 = np.asarray([row[0][0] == row[1][0] for row in collaborative_rows], dtype=np.float64)
        # prefix_lengths 是两部电影从第一级开始连续共享的码数。
        prefix_lengths = np.asarray([
            next((level for level, (left, right) in enumerate(zip(row[0], row[1])) if left != right), len(row[0]))
            for row in collaborative_rows
        ], dtype=np.float64)
        if np.std(pair_weights) > 0 and np.std(prefix_lengths) > 0:
            # Spearman 相关系数比较“共现强度”和“共同 SID 前缀长度”的排序关系。
            correlation = spearmanr(pair_weights, prefix_lengths).statistic
        else:
            correlation = 0.0
        collaborative_quality = {
            "shared_l1_rate": float(shared_l1.mean()),
            "shared_l1_l2_rate": float(np.mean([np.array_equal(row[0][:2], row[1][:2]) for row in collaborative_rows])),
            "cooccurrence_prefix_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        }
    else:
        collaborative_quality = {"shared_l1_rate": 0.0, "shared_l1_l2_rate": 0.0, "cooccurrence_prefix_spearman": 0.0}
    # 报告同时覆盖重建、邻居保持、码本利用、碰撞、类型与协同关系。
    metrics = {
        # 重建向量与原内容向量的逐元素均方误差和平均余弦相似度。
        "reconstruction_mse": float(np.mean((embeddings - reconstructed) ** 2)),
        "reconstruction_cosine": float(np.mean(np.sum(
            embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
            * reconstructed / np.maximum(np.linalg.norm(reconstructed, axis=1, keepdims=True), 1e-12), axis=1,
        ))),
        "neighbor_recall_at_10": _neighbor_recall(embeddings, quantized, min(10, len(embeddings) - 1)),
        "codebooks": _codebook_stats(codes, int(config["rqvae"]["codebook_size"])),
        # unique_semantic_ids 只看原始 L 级码；最终唯一性还要看 C_ 碰撞 token。
        "unique_semantic_ids": len(buckets),
        "collision_rate": collision_movies / len(movie_ids),
        "average_bucket_size": len(movie_ids) / len(buckets),
        "max_bucket_size": max(len(indices) for indices in buckets.values()),
        "unique_mapping_rate_after_collision_token": len(set(tuple(value["tokens"]) for value in mapping.values())) / len(mapping),
        "genre_jaccard": _genre_jaccard(movies, codes, int(config["seed"])),
        "collaborative_quality": collaborative_quality,
    }
    write_json(output_dir / "quality_metrics.json", metrics)
    return metrics


def create_alternative_sid(config: dict[str, Any], mode: str) -> Path:
    """生成 random 或 direct 对照 SID，不使用 RQ-VAE 量化结果。"""
    if mode not in {"random", "direct"}:
        raise ValueError("mode must be random or direct")
    seed_everything(int(config["seed"]))
    artifact_root = Path(config["paths"]["artifact_dir"])
    movie_ids = np.load(artifact_root / "content" / "movie_ids.npy").astype(np.int64)
    output_dir = ensure_dir(artifact_root / f"sid_{mode}")
    mapping: dict[str, Any] = {}
    if mode == "direct":
        # 直接把原 movie_id 变成一个 token，作为不共享语义码的对照。
        for movie_id in movie_ids:
            mapping[str(int(movie_id))] = {"codes": [int(movie_id)], "collision": 0, "tokens": [f"<MOVIE_{int(movie_id)}>"]}
    else:
        levels = int(config["rqvae"]["num_codebooks"])
        size = int(config["rqvae"]["codebook_size"])
        # L 级、每级 K 个码，共有 K^L 种不同组合；不足以覆盖电影时不能唯一分配。
        capacity = size ** levels
        if len(movie_ids) > capacity:
            raise ValueError("Random SID capacity is smaller than the item catalog")
        rng = np.random.default_rng(int(config["seed"]))
        # 无放回抽取组合编号，确保不同电影不会得到相同随机码。
        flat_codes = rng.choice(capacity, size=len(movie_ids), replace=False)
        for movie_id, flat in zip(movie_ids, flat_codes):
            code: list[int] = []
            value = int(flat)
            # 将 0..K^L-1 的整数逐位拆成 K 进制，得到 L 个码编号。
            for _ in range(levels):
                code.append(value % size)
                value //= size
            mapping[str(int(movie_id))] = {
                "codes": code, "collision": 0,
                "tokens": [f"<L{level + 1}_{number}>" for level, number in enumerate(code)] + ["<C_0>"],
            }
    # 两种对照方案与正常 SID 使用相同 JSON 结构，方便后续复用数据集代码。
    path = output_dir / "movie_to_sid.json"
    write_json(path, mapping)
    return path


def export_rqkmeans_sid(config: dict[str, Any]) -> dict[str, Any]:
    """Cluster existing content vectors and export a collision-resolved SID baseline."""
    seed_everything(int(config["seed"]))
    artifact_root = Path(config["paths"]["artifact_dir"])
    content_dir = artifact_root / "content"
    movie_ids = np.load(content_dir / "movie_ids.npy").astype(np.int64)
    embeddings = np.load(content_dir / "movie_embeddings.npy").astype(np.float32)
    device = resolve_device(config.get("device", "auto"))
    vectors = torch.from_numpy(embeddings).to(device)
    rq_cfg = config["rqvae"]
    size = int(rq_cfg["codebook_size"])
    levels = int(rq_cfg["num_codebooks"])
    result = fit_rq_kmeans(
        vectors, [size] * levels,
        iterations=int(rq_cfg.get("kmeans_iterations", 30)),
        seed=int(config["seed"]),
    )
    codes, quantized = encode_rq_kmeans(vectors, result.codebooks)
    codes_np = codes.cpu().numpy()
    quantized_np = quantized.cpu().numpy()
    output_dir = ensure_dir(artifact_root / "sid_rqkmeans")
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for movie_id, code in zip(movie_ids, codes_np):
        buckets[tuple(int(value) for value in code)].append(int(movie_id))
    collision = {movie_id: index for ids in buckets.values() for index, movie_id in enumerate(sorted(ids))}
    mapping = {
        str(int(movie_id)): {
            "codes": [int(value) for value in code],
            "collision": collision[int(movie_id)],
            "tokens": [f"<L{level + 1}_{int(value)}>" for level, value in enumerate(code)]
            + [f"<C_{collision[int(movie_id)]}>"],
        }
        for movie_id, code in zip(movie_ids, codes_np)
    }
    write_json(output_dir / "movie_to_sid.json", mapping)
    np.save(output_dir / "codes.npy", codes_np)
    np.savez_compressed(output_dir / "codebooks.npz", **{
        f"codebook_{level}": centers.cpu().numpy() for level, centers in enumerate(result.codebooks)
    })
    collision_movies = sum(len(ids) for ids in buckets.values() if len(ids) > 1)
    metrics = {
        "method": "rqkmeans",
        "input_space": "content_embeddings",
        "quantization_mse": float(np.mean((embeddings - quantized_np) ** 2)),
        "quantization_cosine": float(np.mean(np.sum(
            embeddings * quantized_np, axis=1
        ) / np.maximum(np.linalg.norm(embeddings, axis=1) * np.linalg.norm(quantized_np, axis=1), 1e-12))),
        "neighbor_recall_at_10": _neighbor_recall(embeddings, quantized_np, min(10, len(embeddings) - 1)),
        "codebooks": _codebook_stats(codes_np, size),
        "unique_semantic_ids": len(buckets),
        "collision_rate": collision_movies / len(movie_ids),
        "max_bucket_size": max(map(len, buckets.values())),
        "unique_mapping_rate_after_collision_token": len({tuple(v["tokens"]) for v in mapping.values()}) / len(movie_ids),
    }
    write_json(output_dir / "quality_metrics.json", metrics)
    return metrics


def export_opq_sid(config: dict[str, Any]) -> dict[str, Any]:
    """Fit OPQ on content vectors and export collision-resolved Semantic IDs."""
    seed_everything(int(config["seed"]))
    artifact_root = Path(config["paths"]["artifact_dir"])
    content_dir = artifact_root / "content"
    movie_ids = np.load(content_dir / "movie_ids.npy").astype(np.int64)
    embeddings = np.load(content_dir / "movie_embeddings.npy").astype(np.float32)
    if len(movie_ids) != len(embeddings) or len(set(map(int, movie_ids))) != len(movie_ids):
        raise ValueError("content movie IDs and embeddings must align uniquely")
    device = resolve_device(config.get("device", "auto"))
    opq_cfg = config.get("opq", {})
    rq_cfg = config["rqvae"]
    levels = int(opq_cfg.get("num_codebooks", rq_cfg["num_codebooks"]))
    size = int(opq_cfg.get("codebook_size", rq_cfg["codebook_size"]))
    result = fit_opq(
        torch.from_numpy(embeddings).to(device), levels, size,
        rotation_iterations=int(opq_cfg.get("rotation_iterations", 10)),
        kmeans_iterations=int(opq_cfg.get("kmeans_iterations", rq_cfg.get("kmeans_iterations", 30))),
        batch_size=int(opq_cfg.get("batch_size", 4096)),
        seed=int(config["seed"]),
    )
    codes = result.codes.cpu().numpy()
    reconstructed = result.reconstruction.cpu().numpy()
    output_dir = ensure_dir(artifact_root / "sid_opq")
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for movie_id, code in zip(movie_ids, codes):
        buckets[tuple(map(int, code))].append(int(movie_id))
    collision = {movie_id: index for ids in buckets.values() for index, movie_id in enumerate(sorted(ids))}
    mapping = {
        str(int(movie_id)): {
            "codes": [int(value) for value in code],
            "collision": collision[int(movie_id)],
            "tokens": [f"<L{level + 1}_{int(value)}>" for level, value in enumerate(code)]
            + [f"<C_{collision[int(movie_id)]}>"],
        }
        for movie_id, code in zip(movie_ids, codes)
    }
    write_json(output_dir / "movie_to_sid.json", mapping)
    np.save(output_dir / "codes.npy", codes)
    np.savez_compressed(
        output_dir / "model.npz",
        rotation=result.rotation.cpu().numpy(),
        **{f"codebook_{level}": table.cpu().numpy() for level, table in enumerate(result.codebooks)},
    )
    collision_movies = sum(len(ids) for ids in buckets.values() if len(ids) > 1)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reconstructed, axis=1)
    metrics = {
        "method": "opq",
        "input_space": "content_embeddings",
        "num_codebooks": levels,
        "codebook_size": size,
        "rotation_iterations": int(opq_cfg.get("rotation_iterations", 10)),
        "kmeans_iterations": int(opq_cfg.get("kmeans_iterations", rq_cfg.get("kmeans_iterations", 30))),
        "quantization_mse": float(np.mean((embeddings - reconstructed) ** 2)),
        "quantization_cosine": float(np.mean(np.sum(embeddings * reconstructed, axis=1) / np.maximum(norms, 1e-12))),
        "neighbor_recall_at_10": _neighbor_recall(embeddings, reconstructed, min(10, len(embeddings) - 1)),
        "codebooks": _codebook_stats(codes, size),
        "unique_semantic_ids": len(buckets),
        "collision_rate": collision_movies / len(movie_ids),
        "max_bucket_size": max(map(len, buckets.values())),
        "unique_mapping_rate_after_collision_token": len({tuple(v["tokens"]) for v in mapping.values()}) / len(movie_ids),
    }
    write_json(output_dir / "quality_metrics.json", metrics)
    return metrics
