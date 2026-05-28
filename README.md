# Rotary Target

旋转目标检测课设工程 scaffold，面向 Word 第 7 题：调研旋转角度表达，基于 DOTA v1.0 对 RoITrans、GlidingVertex、CSL、ReDet、GWD、KLD 等算法做统一训练、评测、可视化和报告汇总。

本仓库不包含数据集、预训练权重或训练产物。默认后端是 MMRotate，当前代码先提供统一 CLI、配置、DOTA 校验、dry-run 命令生成、轻量测试和 harness 文档。

---

## 最低硬件需求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GPU，显存 ≥ 8 GB（如 RTX 3060） | RTX 5090（32 GB）/ RTX 4090（24 GB） |
| CPU | 8 核 | 16 核 |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 250 GB 可用空间（DOTA 原图 + 切片 + 输出） | 500 GB NVMe SSD |
| CUDA | CUDA 11.7+ | CUDA 12.8+（Blackwell 架构需 12.8） |
| 系统 | Ubuntu 20.04 / Windows 10+ | Ubuntu 22.04 / 24.04 |

> 单算法单尺度训练（1024×1024 切片，batch_size=2）约占 7-8 GB 显存。6 个算法全部跑完预计需要 3-5 天（单卡 RTX 3090）/ 1-2 天（RTX 5090）。

---

## 环境配置

### Step 1: 创建 conda 环境

```bash
conda create -n rotary python=3.10 -y
conda activate rotary
```

### Step 2: 安装 PyTorch

根据你的 GPU 架构选择对应版本：

```bash
# RTX 5090 (Blackwell) / RTX 50 系列 — 需要 CUDA 12.8 + PyTorch 2.6+
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu128

# RTX 4090 / RTX 40 系列 — CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# RTX 3090 / RTX 30 系列 — CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

> RTX 5090 用户注意：需要 NVIDIA 驱动 ≥ 570，且必须使用 CUDA 12.8 以上。安装前先执行 `nvidia-smi` 确认驱动版本。

### Step 3: 安装 MMRotate 及依赖

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.0.0,<3.4.0"

git clone -b 1.x https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
cd ..
```

> 版本约束说明：mmdet 3.3.0 要求 mmcv < 2.2.0，务必按上述范围安装避免冲突。

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

```bash
pip install -e ".[dev]"
rt doctor --strict
rt algorithms
rt train --experiment configs/experiments/roi_trans_r50_dota1.yaml --dry-run
rt data validate --root ~/autodl-tmp/DOTA
rt report --dry-run
```

真实训练前需设置 `MMROTATE_ROOT`：

```bash
export MMROTATE_ROOT=/path/to/mmrotate
rt train --experiment configs/experiments/roi_trans_r50_dota1.yaml
```

---

## 完整实验流程

### 1. 数据准备

从 [DOTA 官网](https://captain-whu.github.io/DOTA/dataset.html) 下载 DOTA v1.0 数据集：

| 文件 | 下载地址 |
|------|----------|
| 训练集图像 | https://captain-whu.github.io/DOTA/dataset.html （Google Drive / Baidu Pan） |
| 训练集标注 | 同上页面，选择 Train set labelTxt |
| 验证集图像 | 同上页面，选择 Val set images |
| 验证集标注 | 同上页面，选择 Val set labelTxt |
| 测试集图像 | 同上页面，选择 Test set images |

> 百度网盘备用链接：https://pan.baidu.com/s/1BtrWnAFHkiVFipNn7FAGhA （见官网页面获取提取码）
>
> Google Drive 备用：https://drive.google.com/drive/folders/1UdlgJz1uEhKsValSQ_i3mrLgBIhgojam

**AutoDL 用户**：`/autodl-pub/DOTA` 为只读共享盘，需将标注解压到数据盘：

```bash
mkdir -p ~/autodl-tmp/DOTA/{train,val,test}

# 图片软链接（不额外占空间）
ln -s /root/autodl-pub/DOTA/train/images ~/autodl-tmp/DOTA/train/images
ln -s /root/autodl-pub/DOTA/val/images ~/autodl-tmp/DOTA/val/images
ln -s /root/autodl-pub/DOTA/test/images ~/autodl-tmp/DOTA/test/images 2>/dev/null

# 标注解压到可写数据盘
mkdir -p ~/autodl-tmp/DOTA/train/labelTxt-v1.0
unzip /root/autodl-pub/DOTA/train/labelTxt-v1.0/labelTxt.zip -d ~/autodl-tmp/DOTA/train/labelTxt-v1.0/
mkdir -p ~/autodl-tmp/DOTA/val/labelTxt-v1.0
unzip /root/autodl-pub/DOTA/val/labelTxt-v1.0/labelTxt.zip -d ~/autodl-tmp/DOTA/val/labelTxt-v1.0/
```

最终目录结构：

```
~/autodl-tmp/DOTA/
├── train/
│   ├── images/        -> /root/autodl-pub/DOTA/train/images (symlink)
│   └── labelTxt-v1.0/ (解压后的 .txt 文件)
├── val/
│   ├── images/        -> /root/autodl-pub/DOTA/val/images (symlink)
│   └── labelTxt-v1.0/
└── test/
    └── images/        -> /root/autodl-pub/DOTA/test/images (symlink)
```

验证数据完整性：

```bash
rt data validate --root ~/autodl-tmp/DOTA
```

### 2. 图像切片

DOTA 原图尺寸过大（最大 4000×4000），需切成 1024×1024 的子图：

```bash
export MMROTATE_ROOT=/path/to/mmrotate
rt data split-dota --raw-root ~/autodl-tmp/DOTA \
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
