"""SwinIR气泡重建训练脚本（遮挡程度渐进式训练 + 完整评估）

基于SwinIR架构训练气泡簇到单气泡的图像重建模型，采用课程学习策略。

使用方法:
    python train_bubble_swinir.py \
        --train_root /path/to/dataset/train \
        --val_root /path/to/dataset/val \
        --test_root /path/to/dataset/test \
        --save_dir ./experiments/bubble_swinir \
        --img_size 128 \
        --batch_size 16 \
        --epochs 500 \
        --curriculum_interval 10 \
        --eval_interval 5 \
        --gpu_ids "2,3"

特性:
    - **遮挡程度渐进式训练（课程学习）**：从低遮挡开始逐步增加难度
    - **定期测试评估**：每隔一定epoch在测试集上评估
    - **完整指标**：L1, MSE, PSNR, SSIM
    - **时间戳管理**：自动创建带时间戳的实验文件夹
    - **定期保存**：每隔一定epoch保存checkpoint和可视化
    - TensorBoard日志记录
"""
import argparse
import datetime
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 隐藏INFO级别日志
# ============ 重要：在导入torch之前设置GPU ============
# 必须在导入torch之前设置CUDA_VISIBLE_DEVICES，否则无效
def setup_gpu_before_import():
    """在导入torch之前设置GPU环境变量"""
    # 手动解析--gpu_ids参数（不使用argparse，因为它太重）
    gpu_ids = None
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu_ids' and i + 1 < len(sys.argv):
            gpu_ids = sys.argv[i + 1]
            break

    # 设置环境变量
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        print(f"[GPU Setup] 设置CUDA_VISIBLE_DEVICES={gpu_ids}")
    else:
        # 默认值
        default_gpu_ids = "2,3"
        os.environ["CUDA_VISIBLE_DEVICES"] = default_gpu_ids
        print(f"[GPU Setup] 使用默认GPU配置: {default_gpu_ids}")

# 在导入torch之前执行
setup_gpu_before_import()
# ============ GPU设置完成 ============

import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bubble_swinir_dataset import BubbleSwinIRDataset
from models.network_swinir import SwinIR


# 配置logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def curriculum_categories(available_categories: Sequence[str], epoch: int, interval: int = 10) -> List[str]:
    """课程学习：根据epoch计算当前应使用的类别

    Args:
        available_categories: 所有可用类别（按遮挡程度从低到高排序）
        epoch: 当前epoch（从1开始）
        interval: 每隔多少个epoch增加一个类别

    Returns:
        当前应使用的类别列表
    """
    if not available_categories:
        return []
    stage = min((epoch - 1) // interval + 1, len(available_categories))
    return list(available_categories[:stage])


def format_training_status(
    stage: str,
    epoch: int,
    total_epochs: int,
    batch_idx: int,
    total_batches: int,
    sections: Sequence[Tuple[str, Sequence[str]]],
    batch_time: float,
    eta: datetime.timedelta,
) -> str:
    """构建多行状态字符串以显示训练指标

    Args:
        stage: 阶段名称（'Train', 'Val', 'Test'）
        epoch: 当前epoch
        total_epochs: 总epoch数
        batch_idx: 当前batch索引
        total_batches: 总batch数
        sections: 指标分组，如 [('Loss', ['L1 0.123']), ('Quality', ['PSNR 25.3'])]
        batch_time: 当前batch用时（秒）
        eta: 预计剩余时间

    Returns:
        格式化的状态字符串
    """
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    lines = [f"[{stage}] Epoch {epoch}/{total_epochs} • Batch {batch_idx}/{total_batches} • Time {current_time}"]
    for title, entries in sections:
        if entries:
            lines.append(f"  {title}: " + " | ".join(entries))
    lines.append(f"  Timing: {batch_time:.3f}s/batch | ETA: {eta}")
    return "\n".join(lines)


def plot_metrics(history: Dict[str, List[float]], save_dir: Path, filename: str = "training_curves.png"):
    """绘制训练、验证、测试的损失和指标曲线

    Args:
        history: 指标历史，格式如：
            {
                'epochs': [1, 2, 3, ...],
                'train_loss': [...], 'val_loss': [...], 'test_loss': [...],
                'train_psnr': [...], 'val_psnr': [...], 'test_psnr': [...],
                'train_ssim': [...], 'val_ssim': [...], 'test_ssim': [...],
            }
        save_dir: 保存目录
        filename: 保存文件名
    """
    epochs = history.get('epochs', [])
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # 定义颜色
    colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}

    # 1. Loss曲线
    ax = axes[0, 0]
    for split in ['train', 'val', 'test']:
        key = f'{split}_loss'
        if key in history and history[key]:
            # 过滤掉None值
            valid_data = [(e, v) for e, v in zip(epochs, history[key]) if v is not None]
            if valid_data:
                valid_epochs, valid_values = zip(*valid_data)
                ax.plot(valid_epochs, valid_values, label=split.capitalize(),
                       color=colors[split], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss (L1)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. PSNR曲线
    ax = axes[0, 1]
    for split in ['train', 'val', 'test']:
        key = f'{split}_psnr'
        if key in history and history[key]:
            valid_data = [(e, v) for e, v in zip(epochs, history[key]) if v is not None]
            if valid_data:
                valid_epochs, valid_values = zip(*valid_data)
                ax.plot(valid_epochs, valid_values, label=split.capitalize(),
                       color=colors[split], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR (Peak Signal-to-Noise Ratio)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. SSIM曲线
    ax = axes[1, 0]
    for split in ['train', 'val', 'test']:
        key = f'{split}_ssim'
        if key in history and history[key]:
            valid_data = [(e, v) for e, v in zip(epochs, history[key]) if v is not None]
            if valid_data:
                valid_epochs, valid_values = zip(*valid_data)
                ax.plot(valid_epochs, valid_values, label=split.capitalize(),
                       color=colors[split], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('SSIM (Structural Similarity)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. MSE曲线
    ax = axes[1, 1]
    for split in ['train', 'val', 'test']:
        key = f'{split}_mse'
        if key in history and history[key]:
            valid_data = [(e, v) for e, v in zip(epochs, history[key]) if v is not None]
            if valid_data:
                valid_epochs, valid_values = zip(*valid_data)
                ax.plot(valid_epochs, valid_values, label=split.capitalize(),
                       color=colors[split], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('MSE (Mean Squared Error)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  训练曲线已保存: {save_path}")


class PerceptualLoss(nn.Module):
    """简化的感知损失（使用预训练VGG特征）"""
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision import models
        # 修复警告：使用weights参数替代pretrained
        try:
            # PyTorch 1.13+
            from torchvision.models import VGG16_Weights
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        except ImportError:
            # 兼容旧版本
            vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # 如果是单通道，复制为3通道
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return self.criterion(pred_features, target_features)


class EdgeLoss(nn.Module):
    """边缘损失：提取轮廓并计算MSE

    这个损失函数通过提取图像的边缘（轮廓），然后计算预测图像和目标图像边缘之间的MSE，
    来强制模型生成清晰的单气泡轮廓。
    """
    def __init__(self, edge_method='canny'):
        super().__init__()
        self.edge_method = edge_method
        self.criterion = nn.MSELoss()

        # Sobel算子（用于计算梯度）
        if edge_method == 'sobel':
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
            self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def extract_edges_sobel(self, img):
        """使用Sobel算子提取边缘（可微分）"""
        # img: (B, C, H, W)
        gx = torch.nn.functional.conv2d(img, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(img, self.sobel_y, padding=1)
        # ⭐ 添加epsilon防止sqrt(0)导致梯度爆炸/NaN
        eps = 1e-6
        edges = torch.sqrt(gx ** 2 + gy ** 2 + eps)
        return edges

    def extract_edges_canny(self, img):
        """使用Canny边缘检测（不可微，仅用于可视化）"""
        # 注意：Canny不可微，这里仅作为参考实现
        # 实际训练中建议使用Sobel
        edges = []
        img_np = img.detach().cpu().numpy()
        for i in range(img_np.shape[0]):
            img_uint8 = (np.clip(img_np[i, 0], 0, 1) * 255).astype(np.uint8)
            edge = cv2.Canny(img_uint8, 50, 150)
            edges.append(edge / 255.0)
        edges = np.stack(edges, axis=0)[:, None, :, :]
        return torch.from_numpy(edges).to(img.device).float()

    def forward(self, pred, target):
        """计算边缘损失"""
        if self.edge_method == 'sobel':
            pred_edges = self.extract_edges_sobel(pred)
            target_edges = self.extract_edges_sobel(target)
            return self.criterion(pred_edges, target_edges)
        elif self.edge_method == 'canny':
            # Canny不可微，使用no_grad计算目标边缘
            with torch.no_grad():
                target_edges = self.extract_edges_canny(target)
            pred_edges = self.extract_edges_sobel(pred)  # 预测仍用可微的Sobel
            return self.criterion(pred_edges, target_edges)
        else:
            raise ValueError(f"Unknown edge method: {self.edge_method}")


class SSIMLoss(nn.Module):
    """SSIM损失：1 - SSIM

    SSIM (Structural Similarity Index) 衡量两幅图像的结构相似性。
    损失定义为 1 - SSIM，使得SSIM越高，损失越低。
    """
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma=1.5):
        """生成高斯窗口"""
        gauss = torch.tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        """创建2D高斯窗口"""
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2):
        """计算SSIM"""
        channel = img1.size(1)

        if self.window.device != img1.device or self.window.dtype != img1.dtype:
            self.window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)

        mu1 = torch.nn.functional.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        # ⭐ SSIM稳定性常数（防止除零）
        C1 = (0.01) ** 2  # = 0.0001
        C2 = (0.03) ** 2  # = 0.0009

        # ⭐ 计算SSIM，分母添加小epsilon提高数值稳定性
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / (denominator + 1e-8)

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, target):
        """计算SSIM损失 = 1 - SSIM"""
        return 1 - self._ssim(pred, target)


class GradientLoss(nn.Module):
    """梯度损失：保持图像梯度一致性

    通过计算图像在x和y方向的梯度，并最小化预测图像和目标图像梯度之间的差异，
    来保持边缘的锐利度和结构。
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def gradient(self, img):
        """计算图像梯度（x和y方向）"""
        # img: (B, C, H, W)
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]  # 水平梯度
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]  # 垂直梯度
        return grad_x, grad_y

    def forward(self, pred, target):
        """计算梯度损失"""
        pred_grad_x, pred_grad_y = self.gradient(pred)
        target_grad_x, target_grad_y = self.gradient(target)

        loss_x = self.criterion(pred_grad_x, target_grad_x)
        loss_y = self.criterion(pred_grad_y, target_grad_y)

        return loss_x + loss_y


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """计算重建指标：L1, MSE, PSNR, SSIM

    Args:
        pred: 预测图像 batch, shape=(B, C, H, W)
        target: 目标图像 batch, shape=(B, C, H, W)

    Returns:
        包含各项指标的字典
    """
    # L1
    l1 = torch.mean(torch.abs(pred - target)).item()

    # MSE
    mse = torch.mean((pred - target) ** 2).item()

    # PSNR
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10.0 * math.log10(1.0 / mse)

    # SSIM (需要转为numpy)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    ssim_total = 0.0
    batch_size = pred_np.shape[0]
    for i in range(batch_size):
        pred_img = pred_np[i].squeeze()
        target_img = target_np[i].squeeze()
        ssim_total += structural_similarity(pred_img, target_img, data_range=1.0)
    ssim = ssim_total / batch_size

    return {
        'l1': l1,
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim
    }


def get_optimizer(model: nn.Module, opt) -> optim.Optimizer:
    """创建优化器"""
    return optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)


def get_scheduler(optimizer: optim.Optimizer, opt):
    """创建学习率调度器（warmup + 余弦退火）"""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

    def warmup_lambda(epoch):
        if epoch < opt.warmup_epochs:
            return (epoch + 1) / opt.warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opt.epochs - opt.warmup_epochs,
        eta_min=opt.lr_min
    )

    return warmup_scheduler, cosine_scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_l1: nn.Module,
    criterion_perceptual: PerceptualLoss,
    criterion_edge: EdgeLoss,
    criterion_ssim: SSIMLoss,
    criterion_gradient: GradientLoss,
    opt,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    active_categories: List[str]
) -> Dict[str, float]:
    """训练一个epoch，并实时打印每个iteration的进度"""
    model.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_perceptual = 0.0
    total_edge = 0.0
    total_ssim_loss = 0.0
    total_gradient = 0.0

    # 用于计算ETA
    prev_time = time.time()
    total_batches = len(dataloader)

    # 用于原地更新显示
    num_lines_printed = 0

    for batch_idx, (input_img, target_img, _) in enumerate(dataloader):
        batch_start_time = time.time()

        input_img = input_img.to(device)
        target_img = target_img.to(device)

        # Forward
        pred_img = model(input_img)

        # 计算损失
        loss_l1 = criterion_l1(pred_img, target_img)
        loss = opt.lambda_l1 * loss_l1

        # Perceptual loss
        if opt.lambda_perceptual > 0 and criterion_perceptual is not None:
            loss_perceptual = criterion_perceptual(pred_img, target_img)
            loss += opt.lambda_perceptual * loss_perceptual
            total_perceptual += loss_perceptual.item()

        # Edge loss
        if opt.lambda_edge > 0 and criterion_edge is not None:
            loss_edge = criterion_edge(pred_img, target_img)
            loss += opt.lambda_edge * loss_edge
            total_edge += loss_edge.item()

        # SSIM loss
        if opt.lambda_ssim > 0 and criterion_ssim is not None:
            loss_ssim = criterion_ssim(pred_img, target_img)
            loss += opt.lambda_ssim * loss_ssim
            total_ssim_loss += loss_ssim.item()

        # Gradient loss
        if opt.lambda_gradient > 0 and criterion_gradient is not None:
            loss_gradient = criterion_gradient(pred_img, target_img)
            loss += opt.lambda_gradient * loss_gradient
            total_gradient += loss_gradient.item()

        # ⭐ NaN检测：如果损失为NaN，打印警告并跳过此batch
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  WARNING: NaN/Inf detected in loss at epoch {epoch}, batch {batch_idx}")
            print(f"    Total loss: {loss.item()}")
            print(f"    L1 loss: {loss_l1.item()}")
            if opt.lambda_edge > 0 and criterion_edge is not None:
                print(f"    Edge loss: {loss_edge.item()}")
            if opt.lambda_ssim > 0 and criterion_ssim is not None:
                print(f"    SSIM loss: {loss_ssim.item()}")
            if opt.lambda_gradient > 0 and criterion_gradient is not None:
                print(f"    Gradient loss: {loss_gradient.item()}")
            print(f"    pred_img range: [{pred_img.min():.3f}, {pred_img.max():.3f}]")
            print(f"    target_img range: [{target_img.min():.3f}, {target_img.max():.3f}]")
            print("    Skipping this batch to prevent NaN propagation...")
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # ⭐ 检查梯度是否包含NaN/Inf
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"\n⚠️  WARNING: NaN/Inf detected in gradient of {name}")
                    print(f"    Gradient stats: min={param.grad.min():.3e}, max={param.grad.max():.3e}")
                    has_nan_grad = True

        if has_nan_grad:
            print("    Skipping optimizer step due to NaN gradients...")
            continue

        # 梯度裁剪
        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()

        # 计算评估指标
        with torch.no_grad():
            metrics = compute_metrics(pred_img, target_img)

        # 统计
        batch_size = input_img.size(0)
        total_loss += loss.item() * batch_size
        total_l1 += metrics['l1'] * batch_size
        total_mse += metrics['mse'] * batch_size
        total_psnr += metrics['psnr'] * batch_size
        total_ssim += metrics['ssim'] * batch_size

        # 计算时间和ETA
        current_time = time.time()
        batch_time = current_time - batch_start_time
        batches_done = (epoch - 1) * total_batches + batch_idx
        batches_left = opt.epochs * total_batches - batches_done - 1
        time_left = datetime.timedelta(seconds=batches_left * batch_time)

        # 构建状态信息
        loss_entries = [f'Total {loss.item():.5f}', f'L1 {loss_l1.item():.5f}']
        if opt.lambda_edge > 0 and criterion_edge is not None:
            loss_entries.append(f'Edge {loss_edge.item():.5f}')
        if opt.lambda_ssim > 0 and criterion_ssim is not None:
            loss_entries.append(f'SSIM {loss_ssim.item():.5f}')
        if opt.lambda_gradient > 0 and criterion_gradient is not None:
            loss_entries.append(f'Grad {loss_gradient.item():.5f}')
        if opt.lambda_perceptual > 0 and criterion_perceptual is not None:
            loss_entries.append(f'Percep {loss_perceptual.item():.5f}')

        status = format_training_status(
            stage='Train',
            epoch=epoch,
            total_epochs=opt.epochs,
            batch_idx=batch_idx + 1,
            total_batches=total_batches,
            sections=[
                ('Loss', loss_entries),
                ('Quality', [
                    f'PSNR {metrics["psnr"]:.2f} dB',
                    f'MSE {metrics["mse"]:.5f}',
                    f'SSIM {metrics["ssim"]:.3f}',
                ]),
                ('Categories', [f'{len(active_categories)} active: {", ".join(active_categories[:3])}...' if len(active_categories) > 3 else f'{len(active_categories)} active: {", ".join(active_categories)}']),
            ],
            batch_time=batch_time,
            eta=time_left,
        )

        # 原地更新显示（使用ANSI转义序列）
        if batch_idx > 0:
            # 清除之前打印的内容：上移并清除每一行
            for _ in range(num_lines_printed):
                sys.stdout.write('\033[F')  # 光标上移一行
                sys.stdout.write('\033[K')  # 清除该行

        # 打印新状态
        print(status, flush=True)

        # 记录打印的行数（包括空行）
        num_lines_printed = status.count('\n') + 1

        # 记录到TensorBoard（每个iteration）
        global_step = (epoch - 1) * total_batches + batch_idx
        writer.add_scalar('Train_Iter/Loss', loss.item(), global_step)
        writer.add_scalar('Train_Iter/L1', loss_l1.item(), global_step)
        writer.add_scalar('Train_Iter/PSNR', metrics['psnr'], global_step)
        writer.add_scalar('Train_Iter/SSIM', metrics['ssim'], global_step)
        if opt.lambda_edge > 0 and criterion_edge is not None:
            writer.add_scalar('Train_Iter/EdgeLoss', loss_edge.item(), global_step)
        if opt.lambda_ssim > 0 and criterion_ssim is not None:
            writer.add_scalar('Train_Iter/SSIMLoss', loss_ssim.item(), global_step)
        if opt.lambda_gradient > 0 and criterion_gradient is not None:
            writer.add_scalar('Train_Iter/GradientLoss', loss_gradient.item(), global_step)

    # epoch结束后打印空行，避免被覆盖
    print()

    # 计算平均值（需要除以总样本数）
    total_samples = len(dataloader.dataset)
    result = {
        'loss': total_loss / total_samples,
        'l1': total_l1 / total_samples,
        'mse': total_mse / total_samples,
        'psnr': total_psnr / total_samples,
        'ssim': total_ssim / total_samples,
        'perceptual': total_perceptual / total_batches if opt.lambda_perceptual > 0 else 0.0,
        'edge': total_edge / total_batches if opt.lambda_edge > 0 else 0.0,
        'ssim_loss': total_ssim_loss / total_batches if opt.lambda_ssim > 0 else 0.0,
        'gradient': total_gradient / total_batches if opt.lambda_gradient > 0 else 0.0,
    }
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_l1: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    save_dir: Path,
    dataset_name: str = "val",
    save_samples: bool = False
) -> Dict[str, float]:
    """评估模型（验证集或测试集）

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion_l1: L1损失函数
        device: 设备
        epoch: 当前epoch
        writer: TensorBoard writer
        save_dir: 保存目录
        dataset_name: 数据集名称（"val"或"test"）
        save_samples: 是否保存可视化样本

    Returns:
        包含各项指标的字典
    """
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    # 用于保存样本
    saved_samples = []

    for batch_idx, (input_img, target_img, filenames) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        pred_img = model(input_img)
        loss = criterion_l1(pred_img, target_img)

        # 计算指标
        metrics = compute_metrics(pred_img, target_img)

        total_loss += loss.item()
        total_l1 += metrics['l1']
        total_mse += metrics['mse']
        total_psnr += metrics['psnr']
        total_ssim += metrics['ssim']

        # 保存前几个样本用于可视化
        if save_samples and len(saved_samples) < 8:
            for i in range(min(4, input_img.size(0))):
                if len(saved_samples) >= 8:
                    break
                saved_samples.append({
                    'input': input_img[i].cpu(),
                    'pred': pred_img[i].cpu(),
                    'target': target_img[i].cpu(),
                    'filename': filenames[i]
                })

    n = len(dataloader)
    avg_metrics = {
        'loss': total_loss / n,
        'l1': total_l1 / n,
        'mse': total_mse / n,
        'psnr': total_psnr / n,
        'ssim': total_ssim / n
    }

    # 记录到TensorBoard
    writer.add_scalar(f'{dataset_name.capitalize()}/Loss', avg_metrics['loss'], epoch)
    writer.add_scalar(f'{dataset_name.capitalize()}/L1', avg_metrics['l1'], epoch)
    writer.add_scalar(f'{dataset_name.capitalize()}/MSE', avg_metrics['mse'], epoch)
    writer.add_scalar(f'{dataset_name.capitalize()}/PSNR', avg_metrics['psnr'], epoch)
    writer.add_scalar(f'{dataset_name.capitalize()}/SSIM', avg_metrics['ssim'], epoch)

    # 保存可视化样本
    if save_samples and saved_samples:
        save_visualization_grid(saved_samples, epoch, save_dir, dataset_name)

    return avg_metrics


def save_visualization_grid(samples: List[Dict], epoch: int, save_dir: Path, dataset_name: str):
    """保存可视化结果网格

    Args:
        samples: 样本列表，每个样本包含input/pred/target
        epoch: 当前epoch
        save_dir: 保存目录
        dataset_name: 数据集名称
    """
    vis_dir = save_dir / "visualizations" / f"epoch_{epoch:04d}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        input_np = sample['input'].squeeze().numpy()
        pred_np = sample['pred'].squeeze().numpy()
        target_np = sample['target'].squeeze().numpy()

        # 转换为uint8
        input_np = (np.clip(input_np, 0, 1) * 255).astype(np.uint8)
        pred_np = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
        target_np = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)

        # 拼接图像：输入 | 预测 | 目标
        merged = np.concatenate([input_np, pred_np, target_np], axis=1)

        save_path = vis_dir / f"{dataset_name}_{sample['filename']}"
        cv2.imwrite(str(save_path), merged)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    metrics: Dict[str, float],
    save_dir: Path,
    opt,
    is_best: bool = False
):
    """保存checkpoint

    Args:
        epoch: 当前epoch
        model: 模型（可能是DataParallel包装的）
        optimizer: 优化器
        metrics: 评估指标
        save_dir: 保存目录
        opt: 训练配置
        is_best: 是否是最佳模型
    """
    # 处理DataParallel包装的模型
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
        'config': vars(opt)
    }

    # 保存当前epoch的checkpoint
    save_path = save_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
    torch.save(checkpoint, save_path)
    logger.info(f"  Checkpoint已保存: {save_path}")

    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = save_dir / "checkpoints" / "best_model.pth"
        torch.save(checkpoint, best_path)
        logger.info(f"  最佳模型已保存: {best_path} (PSNR: {metrics.get('psnr', 0):.2f}dB)")


def main():
    parser = argparse.ArgumentParser(description='训练SwinIR气泡重建模型（遮挡程度渐进式训练）')

    # 数据集参数
    parser.add_argument('--train_root', type=str, default='/home/yubd/mount/dataset/dataset_overlap/training_dataset20251111', help='训练集根目录（Deepfillv2格式）')
    parser.add_argument('--val_root', type=str, default='/home/yubd/mount/dataset/dataset_overlap/val', help='验证集根目录（Deepfillv2格式）')
    parser.add_argument('--test_root', type=str, default='/home/yubd/mount/dataset/dataset_overlap/test20251117', help='测试集根目录（Deepfillv2格式）')
    parser.add_argument('--img_size', type=int, default=128, help='图像大小')
    parser.add_argument('--in_chans', type=int, default=1, help='输入通道数(1=灰度, 3=RGB)')
    parser.add_argument('--augment', action='store_true', default=False, help='是否数据增强')

    # 课程学习参数
    parser.add_argument('--curriculum_interval', type=int, default=10, help='课程学习：每隔多少epoch增加一个遮挡类别')
    parser.add_argument('--disable_curriculum', action='store_true', help='禁用课程学习，使用所有类别')

    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=48, help='Embedding维度')
    parser.add_argument('--depths', type=int, nargs='+', default=[4, 4, 6, 6], help='每层深度')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[6, 6, 6, 6], help='注意力头数')
    parser.add_argument('--window_size', type=int, default=8, help='窗口大小')
    parser.add_argument('--mlp_ratio', type=float, default=2.0, help='MLP比例')
    parser.add_argument('--residual_scale', type=float, default=0.5, help='特征级残差系数（控制x_first保留比例）：较深层，推荐0.3-0.8')
    parser.add_argument('--shallow_residual_scale', type=float, default=0.1, help='图像级残差系数（控制x保留比例）：最浅层，推荐0.0-0.2')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup轮数')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值(0表示不裁剪, 推荐1.0)')

    # 损失函数权重
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='L1 loss权重')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='Perceptual loss权重')
    parser.add_argument('--lambda_edge', type=float, default=5.0, help='Edge loss权重（推荐5-10）')
    parser.add_argument('--lambda_ssim', type=float, default=1.0, help='SSIM loss权重（推荐1-2）')
    parser.add_argument('--lambda_gradient', type=float, default=2.0, help='Gradient loss权重（推荐2-5）')
    parser.add_argument('--edge_method', type=str, default='sobel', choices=['sobel', 'canny'], help='边缘检测方法')

    # 保存和日志
    parser.add_argument('--save_dir', type=str, default='/home/yubd/mount/codebase/SwinIR/experiments', help='实验根目录（会自动创建时间戳子文件夹）')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='保存checkpoint间隔(epochs)')
    parser.add_argument('--eval_interval', type=int, default=1, help='评估间隔(epochs)：验证集和测试集')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载workers')

    # 多GPU训练
    parser.add_argument('--multi_gpu', type=bool, default=True, help='是否使用多GPU并行训练（nn.DataParallel）')
    parser.add_argument('--gpu_ids', type = str, default = "2,3", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    # 恢复训练
    parser.add_argument('--resume', type=str, default='', help='恢复训练的checkpoint路径')

    opt = parser.parse_args()

    # 创建带时间戳的保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(opt.save_dir) / f"{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    (save_dir / "visualizations").mkdir(exist_ok=True)

    logger.info(f"实验文件夹: {save_dir}")

    # 保存配置
    with open(save_dir / "config.txt", 'w') as f:
        for key, value in vars(opt).items():
            f.write(f"{key}: {value}\n")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据集
    logger.info("加载数据集...")
    train_dataset_full = BubbleSwinIRDataset(
        opt.train_root,
        img_size=opt.img_size,
        augment=opt.augment,
        in_chans=opt.in_chans
    )
    val_dataset = BubbleSwinIRDataset(
        opt.val_root,
        img_size=opt.img_size,
        augment=False,
        in_chans=opt.in_chans
    )
    test_dataset = BubbleSwinIRDataset(
        opt.test_root,
        img_size=opt.img_size,
        augment=False,
        in_chans=opt.in_chans
    )

    logger.info(f"训练集: {len(train_dataset_full)} 样本")
    logger.info(f"  可用类别: {train_dataset_full.available_categories}")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    logger.info(f"测试集: {len(test_dataset)} 样本")

    # 课程学习配置
    if opt.disable_curriculum:
        logger.info("课程学习已禁用，将使用所有类别训练")
    else:
        logger.info(f"课程学习已启用：每{opt.curriculum_interval}个epoch增加一个遮挡类别")
        logger.info(f"  类别顺序: {train_dataset_full.available_categories}")

    # 验证集和测试集DataLoader（固定）
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    # 模型
    logger.info("构建模型...")
    model = SwinIR(
        upscale=1,
        in_chans=opt.in_chans,
        img_size=opt.img_size,
        window_size=opt.window_size,
        img_range=1.0,
        depths=opt.depths,
        embed_dim=opt.embed_dim,
        num_heads=opt.num_heads,
        mlp_ratio=opt.mlp_ratio,
        upsampler='',
        resi_connection='1conv',
        residual_scale=opt.residual_scale,  # 特征级残差系数（控制x_first）
        shallow_residual_scale=opt.shallow_residual_scale  # 图像级残差系数（控制x）
    )

    # 多GPU支持
    gpu_num = torch.cuda.device_count()
    logger.info(f"检测到 {gpu_num} 个可见GPU（基于CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}）")

    if opt.multi_gpu and gpu_num > 1:
        logger.info(f"使用 {gpu_num} 个GPU进行并行训练 (DataParallel)")
        model = nn.DataParallel(model)
        model = model.cuda()
        # 自动调整batch size和num_workers
        effective_gpus = gpu_num
        opt.batch_size *= effective_gpus
        opt.num_workers = min(opt.num_workers * effective_gpus, 32)
        logger.info(f"  Batch size调整为: {opt.batch_size}")
        logger.info(f"  Num workers调整为: {opt.num_workers}")
    else:
        if gpu_num == 0:
            logger.warning("没有可用的GPU，使用CPU训练")
            model = model.cpu()
        else:
            logger.info(f"使用单GPU训练")
            model = model.cuda()
            effective_gpus = 1

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 打印残差配置
    logger.info("\n残差连接配置:")
    logger.info(f"  Feature-level residual (控制x_first): {opt.residual_scale} (较深层)")
    logger.info(f"  Image-level residual (控制x): {opt.shallow_residual_scale} (最浅层)")
    logger.info("  → 越靠近输入输出，残差比例越小，迫使模型学习修复内容")

    # 损失函数
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss(device) if opt.lambda_perceptual > 0 else None
    criterion_edge = EdgeLoss(edge_method=opt.edge_method).to(device) if opt.lambda_edge > 0 else None
    criterion_ssim = SSIMLoss().to(device) if opt.lambda_ssim > 0 else None
    criterion_gradient = GradientLoss().to(device) if opt.lambda_gradient > 0 else None

    # 打印损失函数配置
    logger.info("\n损失函数配置:")
    logger.info(f"  L1 Loss: {opt.lambda_l1}")
    if opt.lambda_perceptual > 0:
        logger.info(f"  Perceptual Loss: {opt.lambda_perceptual}")
    if opt.lambda_edge > 0:
        logger.info(f"  Edge Loss: {opt.lambda_edge} (method: {opt.edge_method})")
    if opt.lambda_ssim > 0:
        logger.info(f"  SSIM Loss: {opt.lambda_ssim}")
    if opt.lambda_gradient > 0:
        logger.info(f"  Gradient Loss: {opt.lambda_gradient}")

    # ⭐ 打印数值稳定性配置
    logger.info("\n数值稳定性保护:")
    logger.info(f"  Gradient Clipping: {'启用' if opt.grad_clip > 0 else '禁用'} (阈值: {opt.grad_clip})")
    logger.info(f"  NaN Detection: 启用 (自动跳过NaN batch)")
    logger.info(f"  EdgeLoss epsilon: 1e-6 (防止sqrt(0)梯度爆炸)")
    logger.info(f"  SSIM epsilon: 1e-8 (防止除零)")
    if opt.grad_clip == 0:
        logger.warning("  ⚠️  建议启用梯度裁剪以提高稳定性: --grad_clip 1.0")

    # 优化器和调度器
    optimizer = get_optimizer(model, opt)
    warmup_scheduler, cosine_scheduler = get_scheduler(optimizer, opt)

    # TensorBoard
    writer = SummaryWriter(save_dir / "logs")

    # 恢复训练
    start_epoch = 1
    best_psnr = 0.0
    if opt.resume:
        logger.info(f"从checkpoint恢复: {opt.resume}")
        checkpoint = torch.load(opt.resume, weights_only=True)
        # 处理DataParallel包装的模型
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint:
            best_psnr = checkpoint['metrics'].get('psnr', 0.0)

    # 初始化指标历史（用于绘制曲线）
    metrics_history = {
        'epochs': [],
        'train_loss': [],
        'train_psnr': [],
        'train_ssim': [],
        'train_mse': [],
        'val_loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'val_mse': [],
        'test_loss': [],
        'test_psnr': [],
        'test_ssim': [],
        'test_mse': [],
    }

    # 训练循环
    logger.info(f"\n开始训练 (Epoch {start_epoch} - {opt.epochs})...")
    logger.info(f"  Checkpoint保存间隔: 每{opt.checkpoint_interval}个epoch")
    logger.info(f"  评估间隔: 每{opt.eval_interval}个epoch")

    for epoch in range(start_epoch, opt.epochs + 1):
        epoch_start = time.time()

        # ============ 课程学习：调整训练数据集 ============
        if opt.disable_curriculum:
            active_categories = train_dataset_full.available_categories
            epoch_train_dataset = train_dataset_full
        else:
            active_categories = curriculum_categories(
                train_dataset_full.available_categories,
                epoch,
                opt.curriculum_interval
            )
            epoch_train_dataset = train_dataset_full.filtered_dataset(active_categories, strict=True)

        train_loader = DataLoader(
            epoch_train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True
        )

        logger.info(f"\nEpoch {epoch}: 使用类别 {active_categories} ({len(epoch_train_dataset)} 样本)")

        # ============ 训练 ============
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion_l1, criterion_perceptual,
            criterion_edge, criterion_ssim, criterion_gradient,
            opt, device, epoch, writer, active_categories
        )

        # ============ 验证（每个epoch都做） ============
        val_metrics = evaluate(
            model, val_loader, criterion_l1, device, epoch, writer, save_dir,
            dataset_name="val", save_samples=False
        )

        # ============ 测试（每隔eval_interval做一次） ============
        test_metrics = None
        if epoch % opt.eval_interval == 0 or epoch == opt.epochs:
            logger.info(f"  执行测试集评估...")
            test_metrics = evaluate(
                model, test_loader, criterion_l1, device, epoch, writer, save_dir,
                dataset_name="test", save_samples=True  # 保存测试集样本
            )

        # ============ 更新学习率 ============
        if epoch <= opt.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('Curriculum/NumCategories', len(active_categories), epoch)
        writer.add_scalar('Curriculum/NumSamples', len(epoch_train_dataset), epoch)

        epoch_time = time.time() - epoch_start

        # ============ 打印信息 ============
        logger.info(f"\nEpoch {epoch}/{opt.epochs} ({epoch_time:.1f}s)")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.6f}, L1: {train_metrics['l1']:.6f}, "
                   f"PSNR: {train_metrics['psnr']:.2f}dB, SSIM: {train_metrics['ssim']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.6f}, L1: {val_metrics['l1']:.6f}, "
                   f"PSNR: {val_metrics['psnr']:.2f}dB, SSIM: {val_metrics['ssim']:.4f}")
        if test_metrics:
            logger.info(f"  Test  - Loss: {test_metrics['loss']:.6f}, L1: {test_metrics['l1']:.6f}, "
                       f"PSNR: {test_metrics['psnr']:.2f}dB, SSIM: {test_metrics['ssim']:.4f}")
        logger.info(f"  LR: {current_lr:.2e}, Active Categories: {active_categories}")

        # ============ 记录指标历史（用于绘图） ============
        metrics_history['epochs'].append(epoch)
        metrics_history['train_loss'].append(train_metrics['loss'])
        metrics_history['train_psnr'].append(train_metrics['psnr'])
        metrics_history['train_ssim'].append(train_metrics['ssim'])
        metrics_history['train_mse'].append(train_metrics['mse'])

        metrics_history['val_loss'].append(val_metrics['loss'])
        metrics_history['val_psnr'].append(val_metrics['psnr'])
        metrics_history['val_ssim'].append(val_metrics['ssim'])
        metrics_history['val_mse'].append(val_metrics['mse'])

        # 测试集指标（可能为None）
        if test_metrics:
            metrics_history['test_loss'].append(test_metrics['loss'])
            metrics_history['test_psnr'].append(test_metrics['psnr'])
            metrics_history['test_ssim'].append(test_metrics['ssim'])
            metrics_history['test_mse'].append(test_metrics['mse'])
        else:
            # 填充None以保持列表长度一致
            metrics_history['test_loss'].append(None)
            metrics_history['test_psnr'].append(None)
            metrics_history['test_ssim'].append(None)
            metrics_history['test_mse'].append(None)

        # ============ 绘制训练曲线 ============
        # 每隔eval_interval或最后一个epoch绘制一次
        if epoch % opt.eval_interval == 0 or epoch == opt.epochs or epoch == start_epoch:
            plot_metrics(metrics_history, save_dir, filename="training_curves.png")

        # ============ 保存checkpoint ============
        is_best = val_metrics['psnr'] > best_psnr
        if is_best:
            best_psnr = val_metrics['psnr']

        # 每隔checkpoint_interval保存，或者是最佳模型，或者是最后一个epoch
        should_save = (epoch % opt.checkpoint_interval == 0) or is_best or (epoch == opt.epochs)

        if should_save:
            # 保存的指标包含验证集和测试集（如果有）
            save_metrics = {
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim'],
                'val_loss': val_metrics['loss'],
                'psnr': val_metrics['psnr'],  # 用于兼容
            }
            if test_metrics:
                save_metrics.update({
                    'test_psnr': test_metrics['psnr'],
                    'test_ssim': test_metrics['ssim'],
                    'test_loss': test_metrics['loss'],
                })

            save_checkpoint(epoch, model, optimizer, save_metrics, save_dir, opt, is_best)

    writer.close()
    logger.info(f"\n训练完成！")
    logger.info(f"  最佳验证集PSNR: {best_psnr:.2f}dB")
    logger.info(f"  结果保存在: {save_dir}")


if __name__ == '__main__':
    main()
