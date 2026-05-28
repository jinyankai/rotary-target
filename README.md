# Rotary Target

旋转目标检测课设工程 scaffold，面向 Word 第 7 题：调研旋转角度表达，基于 DOTA v1.0 对 RoITrans、GlidingVertex、CSL、ReDet、GWD、KLD 等算法做统一训练、评测、可视化和报告汇总。

本仓库不包含数据集、预训练权重或训练产物。默认后端是 MMRotate，当前代码先提供统一 CLI、配置、DOTA 校验、dry-run 命令生成、轻量测试和 harness 文档。

---

## 最低硬件需求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GPU，显存 ≥ 8 GB（如 RTX 3060） | RTX 3090 / 4090（24 GB） |
| CPU | 8 核 | 16 核 |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 250 GB 可用空间（DOTA 原图 + 切片 + 输出） | 500 GB NVMe SSD |
| CUDA | CUDA 11.7+ | CUDA 12.x |
| 系统 | Ubuntu 20.04 / Windows 10+ | Ubuntu 22.04 |

> 单算法单尺度训练（1024×1024 切片，batch_size=2）约占 7-8 GB 显存。6 个算法全部跑完预计需要 3-5 天（单卡 RTX 3090）。

---

## 环境配置

### Step 1: 创建 conda 环境

```bash
conda create -n rotary python=3.10 -y
conda activate rotary
```

### Step 2: 安装 PyTorch（根据 CUDA 版本选择）

```bash
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: 安装 MMRotate 及依赖

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 克隆 MMRotate（1.x 分支，基于 mmdet 2.x）
git clone -b 1.x https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
cd ..
```

### Step 4: 安装本项目

```bash
cd rotary-target
pip install -e ".[dev,yaml]"
```

### Step 5: 验证环境

```bash
rt doctor --strict
# 预期输出：python [ok], torch [ok], mmrotate [ok], cuda [ok]
```

## Quick Start

```powershell
python -m pip install -e .[dev]
rt doctor --strict
rt algorithms
rt train --experiment configs/experiments/roi_trans_r50_dota1.yaml --dry-run
rt data validate --root data/raw/DOTA-v1.0
rt report --dry-run
```

真实训练前需要另外准备 MMRotate 环境，并设置 `MMROTATE_ROOT` 或传入 `--backend-root`：

```powershell
$env:MMROTATE_ROOT="E:\path\to\mmrotate"
rt train --experiment configs/experiments/roi_trans_r50_dota1.yaml
```

---

## 完整实验流程

### 1. 数据准备

从 [DOTA 官网](https://captain-whu.github.io/DOTA/dataset.html) 下载 DOTA v1.0 数据集，解压到以下结构：

```
data/raw/DOTA-v1.0/
├── train/
│   ├── images/
│   └── labelTxt/
├── val/
│   ├── images/
│   └── labelTxt/
└── test/
    └── images/
```

验证数据完整性：

```bash
rt data validate --root data/raw/DOTA-v1.0
```

### 2. 图像切片

DOTA 原图尺寸过大（最大 4000×4000），需切成 1024×1024 的子图：

```bash
export MMROTATE_ROOT=/path/to/mmrotate
rt data split-dota --raw-root data/raw/DOTA-v1.0 \
                   --out-root data/processed/DOTA-v1.0/split_ss_1024_200 \
                   --tile-size 1024 --gap 200
```

### 3. 训练基线算法（6 个）

逐个训练或编写脚本批量执行：

```bash
# 逐个训练
rt train --experiment configs/experiments/roi_trans_r50_dota1.yaml
rt train --experiment configs/experiments/gliding_vertex_r50_dota1.yaml
rt train --experiment configs/experiments/redet_re50_dota1.yaml
rt train --experiment configs/experiments/csl_retinanet_r50_dota1.yaml
rt train --experiment configs/experiments/gwd_retinanet_r50_dota1.yaml
rt train --experiment configs/experiments/kld_retinanet_r50_dota1.yaml
```

每个算法默认训练 12 epoch，单卡 RTX 3090 约需 8-12 小时。

### 4. 训练创新算法

```bash
rt train --experiment configs/experiments/adaptive_gwd_kld_retinanet_r50_dota1.yaml
```

### 5. 评测

```bash
rt eval --experiment configs/experiments/roi_trans_r50_dota1.yaml
rt eval --experiment configs/experiments/gwd_retinanet_r50_dota1.yaml
# ... 对每个算法执行 eval
```

### 6. 集成推理（创新点）

将多个模型的预测结果融合：

```bash
rt ensemble \
  --predictions outputs/mmrotate/gwd_retinanet_r50_dota1/predictions.json \
                outputs/mmrotate/kld_retinanet_r50_dota1/predictions.json \
                outputs/mmrotate/adaptive_gwd_kld_retinanet_r50_dota1/predictions.json \
  --weights 1.0 1.0 1.5 \
  --output outputs/ensemble/fused_predictions.json
```

### 7. 生成汇总报告

```bash
rt report --output reports/experiment_summary.md
```

## Repository Layout

- `src/rotary_target/`: CLI、配置读取、算法适配器、DOTA 数据工具。
- `src/rotary_target/losses/`: 即插即用自适应混合损失模块（Adaptive GWD-KLD）。
- `src/rotary_target/ensemble/`: 多模型集成推理模块（Oriented Box Fusion）。
- `configs/experiments/`: 每个算法一份实验 YAML。
- `configs/mmrotate/`: 自定义 MMRotate 训练配置文件。
- `docs/`: 数据准备、实验协议、存储估算、创新模块说明。
- `tests/`: 无数据、无 GPU、无 MMRotate 也能跑的轻量测试。

## Storage Guidance

DOTA v1.0 全量训练建议至少预留 250 GB；比较舒服的配置是 500 GB NVMe；多尺度切片或长期多算法实验建议 1 TB。详见 `docs/storage.md`。

## Current Boundary

- 不自动下载 DOTA/HRSC2016。
- 不提交 `data/`、`outputs/`、`checkpoints/`、`*.pth`。
- CI 只跑轻量 scaffold 测试，不安装重型 GPU 依赖。

## 创新模块

详见 [`docs/innovations.md`](docs/innovations.md)：
1. **Adaptive GWD-KLD Loss** — 根据目标宽高比自适应混合两种高斯损失，即插即用
2. **Oriented Weighted Box Fusion** — 多模型旋转框集成推理，角度循环加权融合
