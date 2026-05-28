# 创新模块说明文档

本文档介绍旋转目标检测课设中实现的两个即插即用创新模块。

---

## 1. 自适应混合损失函数 (Adaptive GWD-KLD Hybrid Loss)

### 动机

GWD（Gaussian Wasserstein Distance）和 KLD（Kullback-Leibler Divergence）是旋转目标检测中两种主流的基于高斯分布建模的损失函数。我们观察到：

- **GWD** 对细长目标（高宽比大）更友好：Wasserstein距离在分布形状差异大时梯度平滑，不易出现训练不稳定
- **KLD** 对方形目标更精准：KL散度对分布中心和形状的匹配更敏感，适合宽高比接近1的目标

单独使用任一损失都无法在所有目标形状上达到最优。

### 方法

我们提出根据预测框的宽高比自适应地混合两种损失：

```
L_hybrid = α(r) · L_GWD + (1 - α(r)) · L_KLD

α(r) = sigmoid(k · (r - r₀))
r = max(w, h) / min(w, h)
```

其中 `k=2.0`（平滑系数），`r₀=3.0`（交叉点）为超参数。

**OBB 到高斯分布的转换**：

```
μ = [cx, cy]
Σ = R(θ) · diag(w²/12, h²/12) · R(θ)ᵀ
```

### 即插即用使用方式

在 MMRotate 配置文件中只需两处修改：

```python
custom_imports = dict(
    imports=['rotary_target.losses._torch'],
    allow_failed_imports=False,
)

model = dict(
    bbox_head=dict(
        loss_bbox=dict(type='AdaptiveGWDKLDLoss', k=2.0, r0=3.0, tau=1.0)
    )
)
```

### 文件结构

```
src/rotary_target/losses/
├── __init__.py              # 公开接口
├── adaptive_gwd_kld.py     # 核心算法（纯 numpy，可独立测试）
└── _torch.py               # PyTorch nn.Module 包装 + MMRotate 注册
```

---

## 2. 多模型集成推理 (Oriented Weighted Box Fusion)

### 动机

不同角度编码方式（le90、oc）和不同损失函数训练出的模型具有互补性。例如：
- GWD 模型对大宽高比目标（如 bridge、large-vehicle）检测更稳定
- KLD 模型对紧凑目标（如 roundabout、storage-tank）定位更精准
- CSL 模型的角度分类方式对边界角度更鲁棒

通过在推理阶段融合多个模型的预测，可以获得比任何单一模型更高的 mAP。

### 方法

**Oriented Weighted Box Fusion (OWBF)**：

1. 收集 N 个模型的预测结果
2. 按类别分组，利用旋转 IoU 将重叠框聚类
3. 聚类内加权融合：
   - 位置和尺寸：按 `model_weight × score` 加权平均
   - **角度融合**（关键）：使用循环加权平均处理角度的 π 周期性
     ```
     θ_fused = atan2(Σ wᵢ·sin(2θᵢ), Σ wᵢ·cos(2θᵢ)) / 2
     ```
4. 得分惩罚：`score_fused *= min(matched, N) / N`（抑制单模型假阳性）

**旋转 IoU 计算**：使用 Sutherland-Hodgman 多边形裁剪算法，纯 Python 实现，无外部依赖。

### CLI 使用方式

```bash
# 融合三个模型的预测结果
rt ensemble \
  --predictions outputs/mmrotate/gwd_*/predictions.json \
                outputs/mmrotate/kld_*/predictions.json \
                outputs/mmrotate/adaptive_*/predictions.json \
  --weights 1.0 1.0 1.5 \
  --iou-threshold 0.5 \
  --output outputs/ensemble/fused_predictions.json
```

### 文件结构

```
src/rotary_target/ensemble/
├── __init__.py     # 公开接口
├── geometry.py     # 旋转框几何（corners、polygon intersection、rotated IoU）
├── fusion.py       # OrientedBoxFusion 核心融合逻辑
└── io.py           # 预测结果 JSON 读写
```

---

## 3. 实验配置

| 实验 | 配置文件 | 说明 |
|------|----------|------|
| Adaptive Loss 训练 | `configs/experiments/adaptive_gwd_kld_retinanet_r50_dota1.yaml` | 使用自适应混合损失训练 |
| 集成推理 | `configs/experiments/ensemble_gwd_kld_adaptive.yaml` | 融合 GWD+KLD+Adaptive 三模型 |

### 预期对比维度

| 指标 | GWD | KLD | Adaptive (ours) | Ensemble (ours) |
|------|-----|-----|-----------------|-----------------|
| mAP  | baseline | baseline | ≥GWD/KLD | >all single |
| 大宽高比目标 AP | 优 | 一般 | 优 | 最优 |
| 方形目标 AP | 一般 | 优 | 优 | 最优 |

---

## 4. 运行验证

```bash
# 安装
pip install -e .[dev]

# 测试（无需 GPU/数据/MMRotate）
pytest tests/ -v

# Dry-run 验证
rt train --experiment configs/experiments/adaptive_gwd_kld_retinanet_r50_dota1.yaml --dry-run
rt ensemble --predictions fake1.json fake2.json --dry-run
rt algorithms
```
