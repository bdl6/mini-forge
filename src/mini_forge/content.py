from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import ensure_dir, resolve_device, write_json


def _hashing_embeddings(texts: list[str], dimension: int) -> np.ndarray:
    """把文本映射到固定维度的向量；用于无需下载模型的 smoke 流程。

    这里没有可训练参数。相同文本会得到相同向量，但不同词可能落在同一维。
    """
    # 一行对应一部电影，一列对应哈希后的一个向量维度。
    matrix = np.zeros((len(texts), dimension), dtype=np.float32)
    for row, text in enumerate(texts):
        # 统一大小写，并把类型分隔符 | 当作空格后按词切分。
        for token in text.lower().replace("|", " ").split():
            # 稳定哈希决定词落在哪一列，以及该列加 1 还是减 1。
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            matrix[row, value % dimension] += 1.0 if value & 1 else -1.0
    # 逐行归一化为单位长度；下限防止空文本或正负抵消时除以零。
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


@torch.inference_mode()
def _transformer_embeddings(texts: list[str], model_name: str, batch_size: int, max_length: int, device: torch.device) -> np.ndarray:
    """用预训练编码器生成电影内容向量，不对编码器做梯度更新。"""
    from transformers import AutoModel, AutoTokenizer

    # 分词器把文本变成 token ID；模型输出每个 token 的隐藏向量。
    # 首次使用 model_name 时，from_pretrained 可能需要从模型仓库下载文件。
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding movie content"):
        # 分批处理避免一次把所有电影送入模型；padding 对齐批内长度，
        # truncation 将过长文本截到 max_length 个 token。
        encoded = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        # attention_mask 中填充位置为 0；只对真实 token 求平均，
        # 将 [batch, token 数, hidden_dim] 压成每部电影一个向量。
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        # 先归一化，再移回 CPU 转成 NumPy，便于后续保存和 RQ-VAE 读取。
        batches.append(F.normalize(pooled, dim=-1).cpu().numpy().astype(np.float32))
    # 按输入顺序拼回完整矩阵，每行对应 texts 中同位置的一部电影。
    return np.concatenate(batches)


def encode_movies(config: dict[str, Any]) -> dict[str, Any]:
    """读取 prepare 的电影表，生成并保存与电影 ID 一一对应的内容向量。"""
    processed_dir = Path(config["paths"]["processed_dir"])
    artifact_dir = ensure_dir(Path(config["paths"]["artifact_dir"]) / "content")
    movies = pd.read_csv(processed_dir / "movies.csv")
    # 标题和类型拼成一段文本，中间用 [SEP] 标记分隔。
    texts = [f"{title} [SEP] {str(genres).replace('|', ' | ')}" for title, genres in zip(movies.title, movies.genres)]
    content_cfg = config["content"]
    # smoke 配置使用轻量 hashing；正式配置使用指定的预训练 Transformer。
    if content_cfg["encoder"] == "hashing":
        embeddings = _hashing_embeddings(texts, int(content_cfg.get("output_dim", 384)))
    else:
        embeddings = _transformer_embeddings(
            texts, content_cfg["encoder"], int(content_cfg["batch_size"]),
            int(content_cfg["max_length"]), resolve_device(config.get("device", "auto")),
        )
    # 可选的最终归一化让不同电影向量处在可比较的单位长度尺度上。
    if content_cfg.get("normalize", True):
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    # 两个 .npy 文件保持相同行顺序：movie_ids[i] 对应 movie_embeddings[i]。
    np.save(artifact_dir / "movie_embeddings.npy", embeddings.astype(np.float32))
    np.save(artifact_dir / "movie_ids.npy", movies.movie_id.to_numpy(np.int64))
    # 记录编码器名称和矩阵形状，供检查产物时使用。
    metadata = {"encoder": content_cfg["encoder"], "shape": list(embeddings.shape), "normalized": True}
    write_json(artifact_dir / "metadata.json", metadata)
    return metadata
