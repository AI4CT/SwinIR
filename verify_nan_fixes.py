#!/usr/bin/env python3
"""验证NaN修复是否正确应用

检查项：
1. EdgeLoss的sqrt是否添加epsilon
2. SSIM的除法是否添加epsilon
3. 训练循环是否有NaN检测
4. 梯度裁剪是否默认启用
"""

import re
from pathlib import Path


def check_edge_loss_epsilon():
    """检查EdgeLoss是否添加epsilon"""
    print("✓ 检查EdgeLoss的sqrt epsilon...")

    train_file = Path(__file__).parent / "train_bubble_swinir.py"
    content = train_file.read_text()

    # 查找extract_edges_sobel函数
    pattern = r'def extract_edges_sobel\(self, img\):.*?return edges'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("  ❌ 找不到extract_edges_sobel函数")
        return False

    func_code = match.group(0)

    # 检查是否有eps = 1e-6
    if 'eps = 1e-6' not in func_code:
        print("  ❌ 未找到 eps = 1e-6 定义")
        return False

    # 检查sqrt是否使用eps
    if 'torch.sqrt(gx ** 2 + gy ** 2 + eps)' not in func_code:
        print("  ❌ sqrt未使用epsilon")
        return False

    print("  ✅ EdgeLoss epsilon正确添加")
    return True


def check_ssim_epsilon():
    """检查SSIM是否添加epsilon"""
    print("\n✓ 检查SSIM的除法epsilon...")

    train_file = Path(__file__).parent / "train_bubble_swinir.py"
    content = train_file.read_text()

    # 查找_ssim函数
    pattern = r'def _ssim\(self, img1, img2\):.*?return ssim_map'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("  ❌ 找不到_ssim函数")
        return False

    func_code = match.group(0)

    # 检查除法是否添加epsilon
    if '(denominator + 1e-8)' not in func_code:
        print("  ❌ SSIM除法未添加epsilon")
        return False

    # 检查是否有numerator和denominator变量
    if 'numerator =' not in func_code or 'denominator =' not in func_code:
        print("  ❌ 未找到numerator/denominator变量")
        return False

    print("  ✅ SSIM epsilon正确添加")
    return True


def check_nan_detection():
    """检查是否有NaN检测"""
    print("\n✓ 检查NaN检测机制...")

    train_file = Path(__file__).parent / "train_bubble_swinir.py"
    content = train_file.read_text()

    checks_passed = 0

    # 检查损失NaN检测
    if 'torch.isnan(loss) or torch.isinf(loss)' in content:
        print("  ✅ 损失NaN检测已添加")
        checks_passed += 1
    else:
        print("  ❌ 损失NaN检测缺失")

    # 检查梯度NaN检测
    if 'torch.isnan(param.grad).any() or torch.isinf(param.grad).any()' in content:
        print("  ✅ 梯度NaN检测已添加")
        checks_passed += 1
    else:
        print("  ❌ 梯度NaN检测缺失")

    # 检查是否有跳过机制
    if 'Skipping this batch' in content or 'Skipping optimizer step' in content:
        print("  ✅ NaN跳过机制已添加")
        checks_passed += 1
    else:
        print("  ❌ NaN跳过机制缺失")

    return checks_passed == 3


def check_grad_clip_default():
    """检查梯度裁剪默认值"""
    print("\n✓ 检查梯度裁剪默认配置...")

    train_file = Path(__file__).parent / "train_bubble_swinir.py"
    content = train_file.read_text()

    # 查找grad_clip参数定义
    pattern = r"add_argument\(['\"]--grad_clip['\"].*?default=([0-9.]+)"
    match = re.search(pattern, content)

    if not match:
        print("  ❌ 找不到grad_clip参数定义")
        return False

    default_value = float(match.group(1))

    if default_value == 0.0:
        print(f"  ❌ 梯度裁剪默认禁用 (default={default_value})")
        return False
    elif default_value > 0:
        print(f"  ✅ 梯度裁剪默认启用 (default={default_value})")
        return True

    return False


def check_stability_logging():
    """检查是否有数值稳定性日志"""
    print("\n✓ 检查数值稳定性日志...")

    train_file = Path(__file__).parent / "train_bubble_swinir.py"
    content = train_file.read_text()

    if '数值稳定性保护' in content:
        print("  ✅ 数值稳定性日志已添加")
        return True
    else:
        print("  ❌ 数值稳定性日志缺失")
        return False


def main():
    print("=" * 60)
    print("NaN修复验证脚本")
    print("=" * 60)

    results = []

    results.append(check_edge_loss_epsilon())
    results.append(check_ssim_epsilon())
    results.append(check_nan_detection())
    results.append(check_grad_clip_default())
    results.append(check_stability_logging())

    print("\n" + "=" * 60)
    print(f"验证结果: {sum(results)}/{len(results)} 项通过")
    print("=" * 60)

    if all(results):
        print("✅ 所有NaN修复已正确应用！")
        print("\n可以开始训练:")
        print("  python train_bubble_swinir.py --lambda_edge 5.0 --lambda_ssim 1.0 --lambda_gradient 2.0")
        return 0
    else:
        print("❌ 部分修复未正确应用，请检查上述错误")
        return 1


if __name__ == "__main__":
    exit(main())
