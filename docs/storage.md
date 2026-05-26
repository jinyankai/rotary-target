# Storage Plan

The scaffold itself is small. Without data, checkpoints, and training logs, this repository should stay well under 1 GB.

## DOTA v1.0 Full Training

Recommended reservation:

| Item | Conservative space |
| --- | ---: |
| Raw archives plus extracted original DOTA v1.0 | 60-80 GB |
| Single-scale 1024 patch cache with overlap | 100-180 GB |
| Multiple algorithm checkpoints, logs, visualizations, metrics | 30-80 GB |
| Python/conda/CUDA dependency caches | 20-40 GB |

Minimum practical disk: 250 GB.

Comfortable disk: 500 GB NVMe.

Multi-scale splitting, ablations, or long-running comparisons: 1 TB.

## Repository Policy

Large local artifacts belong under ignored paths:

- `data/raw/`
- `data/processed/`
- `outputs/`
- `checkpoints/`
- `work_dirs/`
- `wandb/`

Do not store DOTA, HRSC2016, downloaded MMRotate checkpoints, or generated `.pth` files in git.
