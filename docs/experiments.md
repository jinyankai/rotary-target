# Experiments

Each experiment is described by a YAML file in `configs/experiments/`.

Required fields:

- `experiment_name`
- `adapter`
- `algorithm`
- `dataset.name`
- `dataset.root`
- `dataset.split_root`

Current algorithm keys:

- `roi_trans`
- `gliding_vertex`
- `redet`
- `csl`
- `gwd`
- `kld`

Dry-run a training command:

```powershell
rt train --experiment configs/experiments/kld_retinanet_r50_dota1.yaml --dry-run
```

Real training requires a prepared MMRotate checkout:

```powershell
$env:MMROTATE_ROOT="E:\path\to\mmrotate"
rt train --experiment configs/experiments/kld_retinanet_r50_dota1.yaml
```

Evaluation:

```powershell
rt eval --experiment configs/experiments/kld_retinanet_r50_dota1.yaml --checkpoint outputs/mmrotate/kld_retinanet_r50_dota1/latest.pth --dry-run
```

Report aggregation expects each completed run to write:

```text
outputs/mmrotate/<experiment>/metrics.json
```
