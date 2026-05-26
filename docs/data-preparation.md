# Data Preparation

This scaffold does not download datasets. Prepare DOTA v1.0 manually from the official source, then place it under:

```text
data/raw/DOTA-v1.0/
  train/
    images/
    labelTxt/
  val/
    images/
    labelTxt/
  test/
    images/
```

Run:

```powershell
rt data validate --root data/raw/DOTA-v1.0
```

The raw DOTA label format is:

```text
x1 y1 x2 y2 x3 y3 x4 y4 category difficult
```

For example:

```text
105.0 120.0 170.0 121.0 168.0 180.0 103.0 179.0 small-vehicle 0
```

To prepare a split-cache command for MMRotate:

```powershell
rt data split-dota --raw-root data/raw/DOTA-v1.0 --out-root data/processed/DOTA-v1.0/split_ss_1024_200 --dry-run
```

Without `--dry-run`, this command writes MMRotate `--base-json` config files under the processed data root, then calls `tools/data/dota/split/img_split.py` from `MMROTATE_ROOT`.

The default project convention is single-scale 1024 tiles with 200 overlap. Multi-scale splitting can be added later as a separate processed dataset root.
