# Rotary Target

旋转目标检测课设工程 scaffold，面向 Word 第 7 题：调研旋转角度表达，基于 DOTA v1.0 对 RoITrans、GlidingVertex、CSL、ReDet、GWD、KLD 等算法做统一训练、评测、可视化和报告汇总。

本仓库不包含数据集、预训练权重或训练产物。默认后端是 MMRotate，当前代码先提供统一 CLI、配置、DOTA 校验、dry-run 命令生成、轻量测试和 harness 文档。

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

## Repository Layout

- `src/rotary_target/`: CLI、配置读取、算法适配器、DOTA 数据工具。
- `configs/experiments/`: 每个算法一份实验 YAML。
- `docs/`: 数据准备、实验协议、存储估算、harness 规则。
- `agents/skills/`: 面向 coding agent 的项目内技能说明。
- `tests/`: 无数据、无 GPU、无 MMRotate 也能跑的轻量测试。

## Storage Guidance

DOTA v1.0 全量训练建议至少预留 250 GB；比较舒服的配置是 500 GB NVMe；多尺度切片或长期多算法实验建议 1 TB。详见 `docs/storage.md`。

## Current Boundary

- 不自动下载 DOTA/HRSC2016。
- 不提交 `data/`、`outputs/`、`checkpoints/`、`*.pth`。
- CI 只跑轻量 scaffold 测试，不安装重型 GPU 依赖。
