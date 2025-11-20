"""SwinIR气泡重建测试脚本

使用训练好的SwinIR模型测试气泡重建效果。

使用方法:
    python test_bubble_swinir.py \
        --test_root /path/to/bubble_dataset/test \
        --checkpoint ./experiments/bubble_swinir/checkpoints/best_model.pth \
        --save_dir ./test_results \
        --img_size 128

输出:
    - 重建结果图像（输入|预测|目标的拼接）
    - 评估指标（PSNR, SSIM, L1）
    - 指标统计CSV文件
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from bubble_swinir_dataset import BubbleSwinIRDataset
from models.network_swinir import SwinIR


@torch.no_grad()
def test_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path
) -> List[Dict]:
    """测试模型并保存结果

    Returns:
        List of metrics for each sample
    """
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for input_imgs, target_imgs, filenames in tqdm(dataloader, desc="Testing"):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)

        # 推理
        pred_imgs = model(input_imgs)

        # 逐样本计算指标和保存
        for i in range(input_imgs.size(0)):
            input_img = input_imgs[i]
            pred_img = pred_imgs[i]
            target_img = target_imgs[i]
            filename = filenames[i]

            # 计算指标
            metrics = compute_metrics(pred_img, target_img)
            metrics['filename'] = filename

            all_metrics.append(metrics)

            # 保存可视化
            save_result(input_img, pred_img, target_img, filename, save_dir)

    return all_metrics


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """计算重建指标

    Args:
        pred: 预测图像 tensor, shape=(C, H, W), range=[0, 1]
        target: 目标图像 tensor, shape=(C, H, W), range=[0, 1]

    Returns:
        Dictionary of metrics
    """
    # 转为numpy
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    # L1
    l1 = np.abs(pred_np - target_np).mean()

    # MSE
    mse = np.square(pred_np - target_np).mean()

    # PSNR
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10.0 * math.log10(1.0 / mse)

    # SSIM
    ssim = structural_similarity(pred_np, target_np, data_range=1.0)

    return {
        'l1': l1,
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim
    }


def save_result(
    input_img: torch.Tensor,
    pred_img: torch.Tensor,
    target_img: torch.Tensor,
    filename: str,
    save_dir: Path
):
    """保存重建结果（拼接图像）

    Args:
        input_img: 输入图像 tensor
        pred_img: 预测图像 tensor
        target_img: 目标图像 tensor
        filename: 文件名
        save_dir: 保存目录
    """
    # 转为numpy并归一化到[0, 255]
    input_np = (input_img.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pred_np = (pred_img.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    target_np = (target_img.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # 拼接：输入 | 预测 | 目标
    merged = np.concatenate([input_np, pred_np, target_np], axis=1)

    # 保存
    save_path = save_dir / filename.replace('.png', '_result.png')
    cv2.imwrite(str(save_path), merged)


def save_metrics_csv(metrics: List[Dict], save_path: Path):
    """保存指标到CSV文件"""
    if not metrics:
        return

    fieldnames = ['filename', 'psnr', 'ssim', 'l1', 'mse']

    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in metrics:
            writer.writerow({
                'filename': m['filename'],
                'psnr': f"{m['psnr']:.4f}",
                'ssim': f"{m['ssim']:.4f}",
                'l1': f"{m['l1']:.6f}",
                'mse': f"{m['mse']:.6f}"
            })


def print_summary(metrics: List[Dict]):
    """打印统计摘要"""
    if not metrics:
        print("没有指标数据")
        return

    psnr_values = [m['psnr'] for m in metrics]
    ssim_values = [m['ssim'] for m in metrics]
    l1_values = [m['l1'] for m in metrics]

    print("\n" + "=" * 60)
    print("测试结果统计:")
    print("=" * 60)
    print(f"样本数量: {len(metrics)}")
    print(f"\nPSNR:")
    print(f"  平均值: {np.mean(psnr_values):.4f} dB")
    print(f"  中位数: {np.median(psnr_values):.4f} dB")
    print(f"  最小值: {np.min(psnr_values):.4f} dB")
    print(f"  最大值: {np.max(psnr_values):.4f} dB")
    print(f"\nSSIM:")
    print(f"  平均值: {np.mean(ssim_values):.4f}")
    print(f"  中位数: {np.median(ssim_values):.4f}")
    print(f"  最小值: {np.min(ssim_values):.4f}")
    print(f"  最大值: {np.max(ssim_values):.4f}")
    print(f"\nL1 Loss:")
    print(f"  平均值: {np.mean(l1_values):.6f}")
    print(f"  中位数: {np.median(l1_values):.6f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='测试SwinIR气泡重建模型')

    # 数据集参数
    parser.add_argument('--test_root', type=str, required=True, help='测试集根目录')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--save_dir', type=str, default='./test_results', help='结果保存目录')

    # 模型参数（应与训练时一致）
    parser.add_argument('--img_size', type=int, default=128, help='图像大小')
    parser.add_argument('--in_chans', type=int, default=1, help='输入通道数')
    parser.add_argument('--embed_dim', type=int, default=96, help='Embedding维度')
    parser.add_argument('--depths', type=int, nargs='+', default=[6, 6, 6, 6], help='每层深度')
    parser.add_argument('--num_heads', type=int, nargs='+', default=[6, 6, 6, 6], help='注意力头数')
    parser.add_argument('--window_size', type=int, default=8, help='窗口大小')
    parser.add_argument('--mlp_ratio', type=float, default=2.0, help='MLP比例')

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=1, help='批大小')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载workers')

    opt = parser.parse_args()

    # 创建保存目录
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    print(f"加载测试集: {opt.test_root}")
    test_dataset = BubbleSwinIRDataset(
        opt.test_root,
        img_size=opt.img_size,
        augment=False,
        in_chans=opt.in_chans
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    print(f"测试样本数: {len(test_dataset)}")

    # 构建模型
    print("构建模型...")
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
        resi_connection='1conv'
    ).to(device)

    # 加载权重
    print(f"加载checkpoint: {opt.checkpoint}")
    checkpoint = torch.load(opt.checkpoint, map_location=device)

    # 兼容不同的checkpoint格式
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best PSNR: {checkpoint.get('best_psnr', 'N/A'):.2f} dB")
    else:
        model.load_state_dict(checkpoint)

    # 测试
    print("\n开始测试...")
    metrics = test_model(model, test_loader, device, save_dir)

    # 保存指标
    csv_path = save_dir / "metrics.csv"
    save_metrics_csv(metrics, csv_path)
    print(f"\n指标已保存到: {csv_path}")

    # 打印统计
    print_summary(metrics)

    print(f"\n重建结果已保存到: {save_dir}")


if __name__ == '__main__':
    main()
