# Canonical Checks

Run these from the repository root:

```powershell
python -m pytest
python -m rotary_target doctor --strict
python -m rotary_target train --experiment configs/experiments/roi_trans_r50_dota1.yaml --dry-run
```

Expected behavior before data preparation:

- `doctor --strict` exits successfully and warns that optional training dependencies/data are missing.
- `train --dry-run` prints the MMRotate command without starting training.
- `data validate` fails for a missing dataset root; that is expected until DOTA is prepared.

Heavy manual checks after data and MMRotate are ready:

```powershell
rt data validate --root data/raw/DOTA-v1.0 --max-files 5000
rt data split-dota --raw-root data/raw/DOTA-v1.0 --out-root data/processed/DOTA-v1.0/split_ss_1024_200
rt train --experiment configs/experiments/roi_trans_r50_dota1.yaml
```
