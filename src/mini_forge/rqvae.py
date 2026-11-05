from __future__ import annotations

# math.ceil 在电影数量不足以填满码本时计算需要重复数据的次数。
import math
# Path 用 / 拼接跨平台路径，避免手写 Windows/Linux 路径分隔符。
from pathlib import Path
# Any 用于标注配置字典中不同类型的值。
from typing import Any

# NumPy 负责读取 .npy 文件、维护电影 ID 和按权重抽样电影对。
import numpy as np
# PyTorch 负责张量运算、自动求导和设备（CPU/CUDA）切换。
import torch
# nn 提供可训练网络层；F 提供不带内部参数的损失和归一化函数。
import torch.nn as nn
import torch.nn.functional as F
# DataLoader 将电影向量分批；TensorDataset 把张量包装成可索引的数据集。
from torch.utils.data import DataLoader, TensorDataset
# trange 是带进度条的 range，用于显示训练轮次。
from tqdm import trange

# 项目工具：原子保存模型、学习率计划、建目录、选设备、固定随机种子、写 JSON。
from .utils import atomic_torch_save, cosine_schedule, ensure_dir, resolve_device, seed_everything, write_json


class ResidualQuantizer(nn.Module):
    """用多张码本逐级近似潜向量；每一级只处理前面没有解释掉的残差。

    设输入为 z，三级选中的码本向量分别为 c1、c2、c3，则前向量化结果是
    c1 + c2 + c3，最终残差是 z - c1 - c2 - c3。
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        dimension: int,
        commitment_beta: float = 0.25,
        codebook_update: str = "gradient",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        # 初始化 nn.Module 的内部状态；否则 PyTorch 无法正确登记下面的码本参数。
        super().__init__()
        # 码本数量，即量化级数 L；默认是 3 级。
        self.num_codebooks = num_codebooks
        # 每张码本有多少个候选向量 K；默认每级 64 个。
        self.codebook_size = codebook_size
        # 每个候选向量的维度 D，必须等于输入潜向量的最后一维。
        self.dimension = dimension
        # 控制编码器输出向选中码本向量靠近的损失权重。
        self.commitment_beta = commitment_beta
        if codebook_update not in {"gradient", "ema"}:
            raise ValueError("codebook_update must be 'gradient' or 'ema'")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be between 0 and 1")
        if ema_eps <= 0.0:
            raise ValueError("ema_eps must be positive")
        self.codebook_update = codebook_update
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        # nn.Embedding(K, D) 是一张可训练的 [K, D] 查找表；第 i 行就是编号 i 的码。
        # nn.ModuleList 会把 L 张码本登记为子模块，使参数进入 model.parameters()、
        # 优化器和 checkpoint；普通 Python list 不会完成这种登记。
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dimension) for _ in range(num_codebooks)])
        for codebook in self.codebooks:
            # uniform_ 原地填入均匀分布随机数，先给每张码本一个初值。
            # 正式训练前 initialize_codebooks 还会用 K-means 中心覆盖这些初值。
            nn.init.uniform_(codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
            if codebook_update == "ema":
                # EMA 直接覆盖码本向量，不让 AdamW 同时通过梯度更新它们。
                codebook.weight.requires_grad_(False)
        # buffer 会随模型保存和移动设备，但不会被优化器更新。
        # 初始计数为 1，相当于为每个码加入一个伪样本，未命中的码可保持原值。
        self.register_buffer("ema_counts", None)
        self.register_buffer("ema_sums", None)
        if codebook_update == "ema":
            self.reset_ema_stats()

    @torch.no_grad()
    def reset_ema_stats(self) -> None:
        """在随机或 K-means 初始化码本后，同步 EMA 的计数和向量和。"""
        if self.codebook_update != "ema":
            return
        # initialize_codebooks 在 inference_mode 中调用；统计量要在普通模式下创建，
        # 否则会成为不可在训练阶段原地更新的 inference tensor。
        with torch.inference_mode(False):
            self.ema_counts = self.codebooks[0].weight.new_ones(self.num_codebooks, self.codebook_size)
            self.ema_sums = torch.stack([codebook.weight.detach().clone() for codebook in self.codebooks])

    @torch.no_grad()
    def _update_ema(self, level: int, residual: torch.Tensor, codes: torch.Tensor) -> None:
        """用当前批次分配到各码的残差，更新该级码本的滑动均值。"""
        counts = torch.bincount(codes, minlength=self.codebook_size).to(residual.dtype)
        sums = torch.zeros_like(self.codebooks[level].weight)
        sums.index_add_(0, codes, residual)
        self.ema_counts[level].mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        self.ema_sums[level].mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)
        # 均值 = 平滑后的向量和 / 平滑后的命中次数；eps 防止分母过小。
        updated = self.ema_sums[level] / self.ema_counts[level].clamp_min(self.ema_eps).unsqueeze(-1)
        self.codebooks[level].weight.copy_(updated)

    def forward(self, latent: torch.Tensor, update_ema: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入 latent [B, D]；返回向量 [B, D]、标量损失、码编号 [B, L]。"""
        # 一开始尚未选任何码，待表示的残差就是完整输入 z。
        residual = latent
        # zeros_like 创建与 latent 形状、dtype、设备相同的全零张量。
        # 后面将每级选中的向量加进来，形成 c1 + c2 + ...。
        quantized = torch.zeros_like(latent)
        # 梯度模式每级收集两项损失；EMA 模式每级只保留 commitment loss。
        losses: list[torch.Tensor] = []
        # 收集每一级选中的整数码编号；稍后拼成 [B, L]。
        all_codes: list[torch.Tensor] = []
        for level, codebook in enumerate(self.codebooks):
            # 当前 residual 是 [B, D]，codebook.weight 是 [K, D]。
            # 利用 ||r-c||² = ||r||² + ||c||² - 2(r·c) 计算所有 B×K 个距离。
            # square() 逐元素平方；sum(-1) 沿最后一维 D 求和。
            # keepdim=True 保留末维，得到 [B, 1]，可与 [B, K] 广播相加。
            distances = (
                residual.square().sum(-1, keepdim=True)
                # 码本每行的平方范数是 [K]；unsqueeze(0) 添一维变成 [1, K]。
                # 广播会把 [B, 1] 与 [1, K] 自动扩展为 [B, K]。
                + codebook.weight.square().sum(-1).unsqueeze(0)
                # t() 转置成 [D, K]；@ 是矩阵乘法，结果 [B, K]。
                - 2.0 * residual @ codebook.weight.t()
            )
            # argmin(-1) 为每个样本找到距离最小的列，即本级码编号 [B]。
            codes = distances.argmin(-1)
            # nn.Embedding 的调用相当于 codebook.weight[codes]，取出 [B, D]。
            selected = codebook(codes)
            # detach() 保留数值但切断该侧梯度；mse_loss 计算均方误差。
            # 梯度模式：第一项推动码本靠近残差；EMA 模式改由批次均值更新码本。
            if self.codebook_update == "gradient":
                losses.append(F.mse_loss(selected, residual.detach()))
            # 第二项只推动 residual（最终是编码器输出）靠近 selected。
            # 系数 beta 控制编码器“承诺”贴近选中码的强度。
            losses.append(self.commitment_beta * F.mse_loss(residual, selected.detach()))
            if self.codebook_update == "ema" and self.training and update_ema:
                # 仅训练时更新；验证和导出 SID 时码本必须保持固定。
                self._update_ema(level, residual.detach(), codes.detach())
            # 将当前选中的码向量计入量化总和。
            quantized = quantized + selected
            # 从当前残差中扣除本级贡献；下一张码本只拟合剩余部分。
            # selected.detach() 使后级残差的梯度不会沿此减法回传给前级码本。
            residual = residual - selected.detach()
            # 保存本级的编号；一部电影最终有 L 个整数码。
            all_codes.append(codes)
        # argmin 的离散选码无法传梯度，因此用直通估计（straight-through）。
        # 前向数值：latent + (quantized - latent) = quantized。
        # 反向梯度：括号整体被 detach() 切断，只沿 latent 传给编码器。
        # 梯度模式的码本由第一项量化损失更新；EMA 模式由滑动均值更新。
        straight_through = latent + (quantized - latent).detach()
        # torch.stack(losses) 把所有标量损失堆成一维，再 sum() 得到总损失。
        # stack(all_codes, dim=-1) 把 L 个 [B] 沿末维拼成 [B, L]。
        return straight_through, torch.stack(losses).sum(), torch.stack(all_codes, dim=-1)


class RQVAE(nn.Module):
    """完整的编码器—残差量化器—解码器。

    默认数据流：[B, 384] -> [B, 256] -> [B, 64] -> 三级离散码
    -> [B, 64] 的量化向量 -> [B, 256] -> [B, 384] 的重建向量。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: int,
        commitment_beta: float = 0.25,
        codebook_update: str = "gradient",
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        # 先初始化 nn.Module，后续赋值的层才能被 PyTorch 登记。
        super().__init__()
        # checkpoint 保存这些结构参数；加载时可据此重建相同形状的网络。
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "commitment_beta": commitment_beta,
            "codebook_update": codebook_update,
            "ema_decay": ema_decay,
            "ema_eps": ema_eps,
        }
        # nn.Sequential 会按写入顺序依次调用各层。
        # nn.Linear(a, b) 做可训练的仿射变换，将最后一维从 a 变成 b。
        # GELU 是非线性激活；默认维度 384 -> 256 -> 64。
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim),
        )
        # 量化器输入和码本向量都在 latent_dim 维的潜空间内。
        self.quantizer = ResidualQuantizer(
            num_codebooks, codebook_size, latent_dim, commitment_beta,
            codebook_update, ema_decay, ema_eps,
        )
        # 解码器把量化后的潜向量映射回原始输入维度，用于计算重建误差。
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, values: torch.Tensor, update_ema: bool = True) -> dict[str, torch.Tensor]:
        """一次完整前向传播；values 是一批电影内容向量 [B, input_dim]。"""
        # 调用编码器得到连续潜向量 [B, latent_dim]。
        latent = self.encoder(values)
        # __call__ 会转到量化器的 forward；得到量化向量、损失和各级整数码。
        quantized, vq_loss, codes = self.quantizer(latent, update_ema=update_ema)
        # 用量化后的向量重建输入；训练时与 values 计算均方误差。
        reconstruction = self.decoder(quantized)
        # dict 让训练代码和 SID 导出代码按名字访问所需结果。
        return {"reconstruction": reconstruction, "vq_loss": vq_loss, "codes": codes, "quantized": quantized}

    # 初始化不需要反向传播；装饰器会关闭自动求导并减少临时开销。
    @torch.inference_mode()
    def initialize_codebooks(self, values: torch.Tensor, iterations: int = 30) -> None:
        """对每一级当前残差做 K-means，以聚类中心作为该级码本初值。"""
        # 先把所有电影内容向量编码到潜空间；这里 values 形状为 [N, input_dim]。
        residual = self.encoder(values)
        for codebook in self.quantizer.codebooks:
            # num_embeddings 是 nn.Embedding 的行数 K，即本级聚类中心数量。
            # assignments[i] 表示第 i 部电影被分配给哪个中心。
            centers, assignments = _torch_kmeans(residual, codebook.num_embeddings, iterations)
            # copy_ 原地覆写码本参数，不会替换 Parameter 对象本身。
            codebook.weight.copy_(centers)
            # 减掉本级中心后，下一张码本拟合尚未解释的残差。
            residual = residual - centers[assignments]
        # EMA 统计量必须与 K-means 后的码本一致，不能沿用随机初始化的值。
        self.quantizer.reset_ema_stats()


# K-means 仅用于初始化，装饰器禁止为这些计算建立梯度图。
@torch.inference_mode()
def _torch_kmeans(values: torch.Tensor, clusters: int, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
    """对 [N, D] 的 values 聚成 K=clusters 类，返回中心 [K, D] 与编号 [N]。"""
    if len(values) >= clusters:
        # randperm(N) 给出 0..N-1 的随机排列；取前 K 个样本作初始中心。
        # clone() 创建独立副本，避免后面修改中心时改动原始 values。
        centers = values[torch.randperm(len(values), device=values.device)[:clusters]].clone()
    else:
        # 样本不足 K 个时，用 ceil(K/N) 算出至少需要重复多少遍。
        repeats = math.ceil(clusters / len(values))
        # repeat(repeats, 1) 沿样本维重复，不改变特征维；再截取 K 行。
        centers = values.repeat(repeats, 1)[:clusters].clone()
    # 预先建立长度 N 的整数张量，存每个样本当前所属簇编号。
    assignments = torch.zeros(len(values), dtype=torch.long, device=values.device)
    # 每次迭代分两步：分配样本到最近中心，再用簇内均值更新中心。
    for _ in range(iterations):
        # cdist 计算每个样本到每个中心的欧氏距离，输出 [N, K]。
        distances = torch.cdist(values, centers)
        # 沿中心维寻找最小距离，得到每个样本的簇编号 [N]。
        assignments = distances.argmin(1)
        for index in range(clusters):
            # 布尔索引挑出当前属于第 index 个簇的所有样本。
            members = values[assignments == index]
            if len(members):
                # mean(0) 沿样本维求均值，得到一个 D 维新中心。
                centers[index] = members.mean(0)
            else:
                # 空簇无法求均值；randint 随机选一个样本重新初始化中心。
                centers[index] = values[torch.randint(len(values), (1,), device=values.device)]
    # 返回最后一轮得到的中心及其样本分配，供逐级码本初始化使用。
    return centers, assignments


def _info_nce(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    """输入 M 对电影的两侧量化向量 [M, D]，计算双向批内对比损失。"""
    # F.normalize 沿最后一维除以向量长度，后续点积就相当于余弦相似度。
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    # left @ right.t() 是 [M, M] 的相似度表；第 i 行第 j 列比较 left_i 与 right_j。
    # 同索引 i==j 是抽到的正对，其余列充当本批次的负候选。
    # 温度越小，后续 softmax 对相似度差别越敏感。
    logits = left @ right.t() / temperature
    # arange(M) 生成 [0, 1, ..., M-1]：第 i 行的正确类别是第 i 列。
    labels = torch.arange(len(left), device=left.device)
    # cross_entropy 按行比较正确列与其他列；转置后再做右->左匹配。
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_rqvae(config: dict[str, Any]) -> dict[str, Any]:
    """读取 embed 与 prepare 的产物，训练量化模型并保存最佳权重。

    流程：准备向量与模型 → 初始化码本 → 准备训练/验证数据与电影对
    → 配置优化器 → 逐轮训练和验证 → 保存最优权重与训练摘要。
    """
    # ===== 1. 准备运行环境、内容向量和输出目录 =====
    # 固定 Python/NumPy/PyTorch 随机种子，尽量使划分、抽样和初始化可复现。
    seed_everything(int(config["seed"]))
    # auto 会在 CUDA 可用时选 GPU，否则选 CPU；返回 torch.device。
    device = resolve_device(config.get("device", "auto"))
    # Path 将配置中的字符串路径变成路径对象；/ 用于继续拼接子目录。
    artifact_root = Path(config["paths"]["artifact_dir"])
    content_dir = artifact_root / "content"
    # ensure_dir 创建输出目录（若尚不存在），并返回可继续拼接的路径。
    output_dir = ensure_dir(artifact_root / "rqvae")
    # np.load 读取 .npy；from_numpy 与 NumPy 数组共享 CPU 内存，再转成 float32。
    # embeddings 的第 i 行与 movie_ids[i] 指向同一部电影。
    embeddings = torch.from_numpy(np.load(content_dir / "movie_embeddings.npy")).float()
    # astype(np.int64) 确保电影 ID 为整数，可用于建立映射和张量索引。
    movie_ids = np.load(content_dir / "movie_ids.npy").astype(np.int64)
    # 只取配置里的 rqvae 小节，例如维度、码本数、学习率和训练轮数。
    rq_cfg = config["rqvae"]
    # 第二维是向量维度；模型第一层的 input_dim 必须与它一致。
    if embeddings.shape[1] != int(rq_cfg["input_dim"]):
        raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match rqvae.input_dim={rq_cfg['input_dim']}")
    # ===== 2. 构建 RQ-VAE，并用残差 K-Means 初始化各层码本 =====
    # 字典推导式只挑出 RQVAE 构造函数必需的参数；** 将键值展开为关键字参数。
    # .to(device) 把模型参数移动到 CPU 或选中的 GPU。
    model = RQVAE(**{
        key: rq_cfg[key] for key in ["input_dim", "hidden_dim", "latent_dim", "num_codebooks", "codebook_size", "commitment_beta"]
    }, codebook_update=str(rq_cfg.get("codebook_update", "gradient")),
        ema_decay=float(rq_cfg.get("ema_decay", 0.99)),
        ema_eps=float(rq_cfg.get("ema_eps", 1e-5))).to(device)
    # 先用所有内容向量对码本做 K-means 初始化，然后才开始梯度训练。
    # embeddings.to(device) 产生设备上的张量；原始 embeddings 仍保留在 CPU。
    model.initialize_codebooks(embeddings.to(device), int(rq_cfg.get("kmeans_iterations", 30)))

    # ===== 3. 划分重建训练集与验证集，创建 DataLoader =====
    # 这里是电影向量的重建批次划分，不是 prepare 阶段按用户时间切分的样本。
    # 注意：码本初始化和下方 i2i 对比项仍可能使用验证电影，因此它不是完全隔离的验证集。
    # 单独的 Generator 固定本次划分所用的随机数序列。
    generator = torch.Generator().manual_seed(int(config["seed"]))
    # round 计算验证电影数，max(1, ...) 确保至少留一部电影。
    validation_count = max(1, round(len(embeddings) * float(rq_cfg.get("validation_ratio", 0.1))))
    # randperm 给出电影行号的随机排列，不改动原向量顺序。
    permutation = torch.randperm(len(embeddings), generator=generator)
    # 前 validation_count 个行号用于验证，其余用于重建训练批次。
    validation_indices, train_indices = permutation[:validation_count], permutation[validation_count:]
    # TensorDataset 把向量包装成单字段数据集；DataLoader 按 batch_size 分批。
    # 训练批次每轮打乱，验证批次保持固定顺序。
    train_loader = DataLoader(TensorDataset(embeddings[train_indices]), batch_size=int(rq_cfg["batch_size"]), shuffle=True)
    validation_loader = DataLoader(TensorDataset(embeddings[validation_indices]), batch_size=int(rq_cfg["batch_size"]), shuffle=False)

    # ===== 4. 准备 i2i 电影对及按共现权重抽样的概率 =====
    # i2i_pairs.npy 每行是 [电影A ID, 电影B ID, 累计共现权重]。
    # np.load 读入该 [P, 3] 数组；P 是不同电影对数量。
    raw_pairs = np.load(Path(config["paths"]["processed_dir"]) / "i2i_pairs.npy")
    # 建立电影 ID -> embeddings 行号的映射。
    id_to_index = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    # 将每对电影 ID 转成行号；若某部电影没有内容向量，则忽略该对。
    valid_pairs = [(id_to_index[int(a)], id_to_index[int(b)], float(w)) for a, b, w in raw_pairs if int(a) in id_to_index and int(b) in id_to_index]
    if valid_pairs:
        # 转为数值数组，前两列是电影行号，第三列是共现权重。
        pair_array = np.asarray(valid_pairs, dtype=np.float64)
        # 各权重除以总和，得到总和为 1 的抽样概率。
        # 权重只决定某对被抽中的机会，不直接乘在单个 InfoNCE 损失上。
        pair_probabilities = pair_array[:, 2] / pair_array[:, 2].sum()
    else:
        # 若没有有效电影对，使用空数组；训练循环会跳过 i2i 损失。
        pair_array = np.empty((0, 3), dtype=np.float64)
        pair_probabilities = np.empty(0, dtype=np.float64)

    # ===== 5. 配置优化器、学习率计划和早停状态 =====
    # AdamW 只更新 requires_grad=True 的参数；EMA 模式下码本由滑动均值更新。
    # weight_decay 是优化器的权重衰减，不是下面总损失中的单独一项。
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(rq_cfg["learning_rate"]), weight_decay=float(rq_cfg["weight_decay"]),
    )
    # 每个训练 batch 更新一次参数；按最大轮数估算总优化步数。
    total_steps = int(rq_cfg["epochs"]) * max(1, len(train_loader))
    # 将 warmup_epochs 换算为预热阶段的优化步数。
    warmup_steps = int(rq_cfg["warmup_epochs"]) * max(1, len(train_loader))
    # LambdaLR 每次 step() 调用 cosine_schedule 计算学习率倍率：先预热再余弦下降。
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_schedule(step, total_steps, warmup_steps))
    # 最优验证重建误差从无穷大开始，第一轮必然优于它。
    best_validation = float("inf")
    # 记录连续多少轮没有刷新最优验证误差。
    patience = 0
    # 保存每轮的训练和验证指标，最后写入 history.json。
    history: list[dict[str, float]] = []

    # ===== 6. 逐轮训练：每轮遍历重建 batch，再验证并决定是否保存 =====
    # trange 显示以 epoch 为单位的进度条；最多迭代 epochs 轮。
    for epoch in trange(int(rq_cfg["epochs"]), desc="Training RQ-VAE"):
        # train() 开启训练模式；本模型无 dropout，但此调用仍是标准做法。
        model.train()
        # 累加本轮每个 batch 的四种损失，轮末再取平均。
        totals = {"loss": 0.0, "recon": 0.0, "vq": 0.0, "i2i": 0.0}
        # TensorDataset 只有一个字段，因此 batch 外面的逗号用于单元素元组解包。
        for (batch,) in train_loader:
            # 6.1 重建分支：编码、残差量化、解码并计算重建误差。
            # DataLoader 给的是 CPU 张量；移动到模型所在设备。
            batch = batch.to(device)
            # model(batch) 调用 RQVAE.forward，得到重建向量、量化损失等。
            output = model(batch)
            # MSE 比较重建向量和原内容向量，越小代表保留的内容信息越多。
            reconstruction_loss = F.mse_loss(output["reconstruction"], batch)
            # 6.2 对比分支：按权重抽取关联电影对，计算双向 InfoNCE。
            # 标量 0 放在同一设备；当无有效电影对或 i2i_weight=0 时直接使用。
            contrastive_loss = torch.zeros((), device=device)
            if len(pair_array) and float(rq_cfg["i2i_weight"]) > 0:
                # 每个内容批次都按累计权重抽取电影对；最多抽 pair_batch_size 对。
                sample_count = min(int(rq_cfg["pair_batch_size"]), len(pair_array))
                # choice 的 p 指定各对概率；replace=True 表示同一对可重复抽中。
                sampled = np.random.choice(len(pair_array), size=sample_count, replace=True, p=pair_probabilities)
                # as_tensor 把抽中的行号转换为 PyTorch 的 long（整数索引）张量。
                left_indices = torch.as_tensor(pair_array[sampled, 0], dtype=torch.long)
                right_indices = torch.as_tensor(pair_array[sampled, 1], dtype=torch.long)
                # 用行号找到电影内容向量，经同一个 RQVAE 得到左右两侧的量化向量。
                # 此处只读取码本，不更新 EMA，以免左右两侧使用不同版本的码本。
                # 这些电影对的采样与上面的重建 train_indices 无关，可能包含验证电影。
                left = model(embeddings[left_indices].to(device), update_ema=False)["quantized"]
                right = model(embeddings[right_indices].to(device), update_ema=False)["quantized"]
                # InfoNCE 让抽中的左右电影向量更容易相互匹配。
                contrastive_loss = _info_nce(left, right, float(rq_cfg["temperature"]))
            # 6.3 合并损失，反向传播并更新模型参数。
            # 梯度模式有重建、码本、commitment、i2i；EMA 模式去掉码本 MSE。
            # i2i_weight 只缩放整个对比损失。
            loss = reconstruction_loss + output["vq_loss"] + float(rq_cfg["i2i_weight"]) * contrastive_loss
            # 清掉上一 batch 的梯度；set_to_none=True 让 PyTorch 按需重新分配。
            optimizer.zero_grad(set_to_none=True)
            # backward 沿计算图求各可训练参数的梯度。
            loss.backward()
            # 若总梯度范数超过阈值，就按比例缩小，避免更新过猛。
            nn.utils.clip_grad_norm_(model.parameters(), float(rq_cfg["gradient_clip"]))
            # 用当前梯度更新模型权重；再把学习率计划推进一步。
            optimizer.step()
            scheduler.step()
            # 6.4 累计本轮训练指标，供轮末取平均并写日志。
            # detach 后转为 Python float，避免保存整个反向计算图。
            # 这里记录的 i2i 是未乘 i2i_weight 的原始对比损失。
            for key, value in [("loss", loss), ("recon", reconstruction_loss), ("vq", output["vq_loss"]), ("i2i", contrastive_loss)]:
                totals[key] += float(value.detach())

        # ===== 7. 在验证集上评估重建质量，不更新参数 =====
        # eval() 切换到评估模式；不改变模型参数。
        model.eval()
        # 分别累计验证集的重建 MSE 与余弦相似度。
        validation_recon = 0.0
        validation_cosine = 0.0
        # inference_mode 禁止构建梯度图，节省验证阶段的内存和计算。
        with torch.inference_mode():
            for (batch,) in validation_loader:
                # 验证向量也要移动到模型所在设备，然后做前向计算。
                batch = batch.to(device)
                output = model(batch)
                # float 将单个张量标量变为 Python 数值，便于累计。
                validation_recon += float(F.mse_loss(output["reconstruction"], batch))
                # cosine_similarity 默认沿最后一维比较两向量；mean 对本批电影求平均。
                validation_cosine += float(F.cosine_similarity(output["reconstruction"], batch).mean())
        # 这里按验证 batch 数取平均，不是按电影数加权平均。
        validation_recon /= max(1, len(validation_loader))
        validation_cosine /= max(1, len(validation_loader))
        # 同样按训练 batch 数取平均，形成一轮的日志指标。
        epoch_result = {key: value / max(1, len(train_loader)) for key, value in totals.items()}
        # epoch + 1 把从 0 开始的循环编号改成便于阅读的第 1、2、... 轮。
        epoch_result.update({"epoch": epoch + 1, "validation_recon": validation_recon, "validation_cosine": validation_cosine})
        history.append(epoch_result)
        # ===== 8. 按验证重建误差保存最优 checkpoint，并检查早停 =====
        # 只按 validation_recon 选择最优模型；cosine 和 i2i 值用于观察，不参与选择。
        if validation_recon < best_validation:
            # 刷新最优值，重置“无改善轮数”。
            best_validation = validation_recon
            patience = 0
            # state_dict 包含各层权重；atomic_torch_save 先写临时文件再替换目标。
            atomic_torch_save({"model_config": model.model_config, "state_dict": model.state_dict()}, output_dir / "best.pt")
        else:
            patience += 1
            # 连续达到容忍轮数就提前退出，不再跑满 epochs。
            if patience >= int(rq_cfg["early_stop_patience"]):
                break
    # ===== 9. 保存完整训练历史和摘要 =====
    # 完整训练结束或提前停止后，再统一保存每轮记录和摘要。
    write_json(output_dir / "history.json", history)
    summary = {"best_validation_recon": best_validation, "epochs_trained": len(history), "device": str(device)}
    write_json(output_dir / "train_summary.json", summary)
    return summary


def load_rqvae(path: str | Path, device: torch.device) -> RQVAE:
    """读取训练时保存的 best.pt，重建模型并切换到评估模式。"""
    # torch.load 读取 checkpoint；map_location 允许在不同设备上加载。
    # weights_only=False 会按 PyTorch 常规方式反序列化整个保存对象。
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # ** 将保存的模型结构参数展开，创建形状一致但尚未加载权重的新模型。
    model = RQVAE(**checkpoint["model_config"])
    # load_state_dict 把 checkpoint 中编码器、码本、解码器的权重装回模型。
    model.load_state_dict(checkpoint["state_dict"])
    # 移到目标设备并调用 eval()；export-sid 随后用它做电影 SID 推断。
    return model.to(device).eval()

