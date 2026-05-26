# MMRotate Backend Notes

The project does not vendor MMRotate. Install or clone MMRotate separately, then point this scaffold at it:

```powershell
$env:MMROTATE_ROOT="E:\path\to\mmrotate"
rt train --experiment configs/experiments/roi_trans_r50_dota1.yaml
```

Default upstream config mappings live in `src/rotary_target/algorithms.py`. For branch-specific changes, set `training.config` in an experiment YAML to the exact config path.
