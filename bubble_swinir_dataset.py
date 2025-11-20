"""SwinIR气泡数据集加载器（适配Deepfillv2数据格式）

直接使用Deepfillv2的数据集格式（Blocked/Mask/Origin），支持遮挡程度渐进式训练。

使用方法:
    from bubble_swinir_dataset import BubbleSwinIRDataset

    dataset = BubbleSwinIRDataset(
        root='/path/to/dataset',  # 包含Blocked/Mask/Origin的目录
        img_size=128,
        augment=True,
        allowed_categories=['0-10', '10-20']  # 课程学习：只使用特定遮挡类别
    )

特性:
    - 直接读取Deepfillv2格式数据（Blocked/Origin），不使用Mask
    - 支持category过滤，用于课程学习
    - 支持数据增强
"""

from __future__ import annotations

import functools
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

_SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# 图像缓存
@functools.lru_cache(maxsize=1000)
def _load_cached_image(path: str, imgsize: int, grayscale: bool = True) -> np.ndarray:
    """加载并缓存图像"""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    if img.shape[0] == imgsize and img.shape[1] == imgsize:
        return img
    return cv2.resize(img, (imgsize, imgsize), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class BubbleSample:
    """气泡样本容器"""
    blocked_path: Path
    mask_path: Path
    origin_path: Path
    category: str


def _strip_suffix(name: str, suffix: str) -> str:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _resolve_partner(directory: Path, stem: str, suffix: str) -> Optional[Path]:
    """查找特定后缀的配对文件"""
    for ext in _SUPPORTED_EXTS:
        candidate = directory / f"{stem}{suffix}{ext}"
        if candidate.exists():
            return candidate
    matches = list(directory.glob(f"{stem}{suffix}.*"))
    return matches[0] if matches else None


def _collect_triplets(base_dir: Path, category: Optional[str] = None) -> List[BubbleSample]:
    """收集Blocked/Mask/Origin三元组"""
    blocked_dir = base_dir / "Blocked"
    mask_dir = base_dir / "Mask"
    origin_dir = base_dir / "Origin"

    if not blocked_dir.is_dir() or not mask_dir.is_dir() or not origin_dir.is_dir():
        LOGGER.debug("Skipping directory %s because required sub-folders are missing", base_dir)
        return []

    category = category or base_dir.name
    samples: List[BubbleSample] = []

    for blocked_path in sorted(blocked_dir.glob("*")):
        if not blocked_path.is_file() or blocked_path.suffix.lower() not in _SUPPORTED_EXTS:
            continue

        stem_with_suffix = blocked_path.stem
        base_stem = _strip_suffix(stem_with_suffix, "_blocked")

        mask_path = _resolve_partner(mask_dir, base_stem, "_mask")
        origin_path = _resolve_partner(origin_dir, base_stem, "_original")

        if mask_path is None or origin_path is None:
            LOGGER.warning(
                "Incomplete triplet for %s (mask: %s, origin: %s)",
                blocked_path,
                mask_path,
                origin_path,
            )
            continue

        samples.append(
            BubbleSample(
                blocked_path=blocked_path,
                mask_path=mask_path,
                origin_path=origin_path,
                category=category,
            )
        )

    return samples


def _discover_samples(root: Path) -> List[BubbleSample]:
    """发现所有气泡样本"""
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    samples: List[BubbleSample] = []

    # 两种布局：
    # 1. root直接包含Blocked/Mask/Origin
    # 2. root包含类别文件夹（如0-10/），每个类别文件夹包含Blocked/Mask/Origin

    direct_samples = _collect_triplets(root, category=root.name)
    if direct_samples:
        samples.extend(direct_samples)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"Blocked", "Mask", "Origin", "Segmented"}:
            continue
        samples.extend(_collect_triplets(child, category=child.name))

    return samples


def _sort_categories(categories: Iterable[str]) -> List[str]:
    """对类别排序（按遮挡程度从低到高）"""
    def _key(cat: str) -> Tuple[int, str]:
        try:
            prefix = cat.split('-')[0]
            return int(prefix), cat
        except ValueError:
            return (1 << 30, cat)

    return sorted(set(categories), key=_key)


class BubbleSwinIRDataset(Dataset):
    """SwinIR气泡数据集（Deepfillv2格式，支持课程学习）

    数据格式:
        root/
        ├── 0-10/           # 遮挡类别（可选）
        │   ├── Blocked/
        │   ├── Mask/       # SwinIR不使用，但需要存在
        │   └── Origin/
        ├── 10-20/
        │   ├── Blocked/
        │   ├── Mask/
        │   └── Origin/
        └── ...

    或者扁平结构:
        root/
        ├── Blocked/
        ├── Mask/
        └── Origin/
    """

    def __init__(
        self,
        root: str,
        img_size: int = 128,
        augment: bool = True,
        in_chans: int = 1,
        allowed_categories: Optional[Sequence[str]] = None,
        strict: bool = True,
        samples: Optional[List[BubbleSample]] = None,
    ) -> None:
        """
        Args:
            root: 数据集根目录
            img_size: 图像大小
            augment: 是否数据增强
            in_chans: 输入通道数（1=灰度，3=RGB）
            allowed_categories: 允许的类别列表（用于课程学习）
            strict: 如果为True且没有样本，则抛出异常
            samples: 预先提供的样本列表（用于filtered_dataset）
        """
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment
        self.in_chans = in_chans
        self._allowed_categories = set(allowed_categories) if allowed_categories else None

        # 发现所有样本
        all_samples = list(samples) if samples is not None else _discover_samples(self.root)
        self._all_samples: List[BubbleSample] = all_samples

        # 过滤类别
        if self._allowed_categories is not None:
            working_samples = [s for s in all_samples if s.category in self._allowed_categories]
        else:
            working_samples = list(all_samples)

        if not working_samples:
            message = f"No samples discovered under {self.root}"
            if self._allowed_categories:
                message += f" with categories {sorted(self._allowed_categories)}"
            if strict:
                raise RuntimeError(message)
            LOGGER.warning(message)

        self._samples: List[BubbleSample] = working_samples
        self.samples = self._samples  # 向后兼容

        # 类别信息
        self.available_categories: List[str] = _sort_categories(s.category for s in self._all_samples)
        self.active_categories: List[str] = _sort_categories(s.category for s in self._samples)

        LOGGER.info(f"Loaded dataset from {root}")
        LOGGER.info(f"  Total samples: {len(self._all_samples)}")
        LOGGER.info(f"  Available categories: {self.available_categories}")
        if self._allowed_categories:
            LOGGER.info(f"  Active samples: {len(self._samples)} (filtered by {self.active_categories})")
        else:
            LOGGER.info(f"  Active samples: {len(self._samples)} (all categories)")

    def __len__(self) -> int:
        return len(self._samples)

    def _augment(self, input_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强：同步对输入和目标进行相同的变换"""
        # 随机水平翻转
        if random.random() > 0.5:
            input_img = np.fliplr(input_img).copy()
            target_img = np.fliplr(target_img).copy()

        # 随机垂直翻转
        if random.random() > 0.5:
            input_img = np.flipud(input_img).copy()
            target_img = np.flipud(target_img).copy()

        # 随机90度旋转
        k = random.randint(0, 3)  # 0, 90, 180, 270度
        if k > 0:
            input_img = np.rot90(input_img, k).copy()
            target_img = np.rot90(target_img, k).copy()

        return input_img, target_img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            input_tensor: shape=(C, H, W)，被遮挡的气泡簇图像
            target_tensor: shape=(C, H, W)，完整的单气泡图像
            filename: 文件名（用于保存结果）
        """
        sample = self._samples[index]

        # 加载图像（SwinIR不使用mask）
        grayscale = (self.in_chans == 1)
        blocked = _load_cached_image(str(sample.blocked_path), self.img_size, grayscale)
        origin = _load_cached_image(str(sample.origin_path), self.img_size, grayscale)

        if blocked is None or origin is None:
            raise RuntimeError(f"Failed to read sample: {sample}")

        # 归一化到[0, 1]
        blocked_norm = blocked.astype(np.float32) / 255.0
        origin_norm = origin.astype(np.float32) / 255.0

        # 数据增强
        if self.augment:
            blocked_norm, origin_norm = self._augment(blocked_norm, origin_norm)

        # 转换为tensor
        if self.in_chans == 1:
            blocked_tensor = torch.from_numpy(blocked_norm).unsqueeze(0)  # (1, H, W)
            origin_tensor = torch.from_numpy(origin_norm).unsqueeze(0)
        else:
            # RGB: (H, W, C) -> (C, H, W)
            blocked_tensor = torch.from_numpy(blocked_norm).permute(2, 0, 1)
            origin_tensor = torch.from_numpy(origin_norm).permute(2, 0, 1)

        return blocked_tensor, origin_tensor, sample.blocked_path.name

    def filtered_dataset(self, categories: Sequence[str], strict: bool = True) -> "BubbleSwinIRDataset":
        """创建过滤后的数据集（用于课程学习）

        Args:
            categories: 要包含的类别列表
            strict: 如果为True且没有样本，则抛出异常

        Returns:
            新的过滤后的数据集实例
        """
        allowed = set(categories)
        filtered = [s for s in self._all_samples if s.category in allowed]
        return BubbleSwinIRDataset(
            root=str(self.root),
            img_size=self.img_size,
            augment=self.augment,
            in_chans=self.in_chans,
            allowed_categories=None,  # 已经过滤，不需要再过滤
            strict=strict,
            samples=filtered,
        )


# 快速测试
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 测试数据集加载
    dataset = BubbleSwinIRDataset(
        root="/home/yubd/mount/dataset/dataset_overlap/training_dataset20251111",
        img_size=128,
        augment=True,
        in_chans=1
    )

    print(f"\n数据集统计:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  可用类别: {dataset.available_categories}")
    print(f"  激活类别: {dataset.active_categories}")

    # 测试课程学习
    print(f"\n测试课程学习:")
    for stage in [1, 2, 3, 4]:
        categories = dataset.available_categories[:stage]
        filtered = dataset.filtered_dataset(categories)
        print(f"  Stage {stage} (categories={categories}): {len(filtered)} samples")

    # 可视化第一个样本
    if len(dataset) > 0:
        input_img, target_img, filename = dataset[0]
        print(f"\n样本信息:")
        print(f"  输入shape: {input_img.shape}")
        print(f"  目标shape: {target_img.shape}")
        print(f"  文件名: {filename}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(input_img.squeeze(), cmap='gray')
        axes[0].set_title('Input (Blocked Cluster)')
        axes[0].axis('off')

        axes[1].imshow(target_img.squeeze(), cmap='gray')
        axes[1].set_title('Target (Complete Bubble)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig('/tmp/bubble_swinir_dataset_sample.png', dpi=150)
        print(f"\n样本可视化已保存到 /tmp/bubble_swinir_dataset_sample.png")
