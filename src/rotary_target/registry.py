from __future__ import annotations

from rotary_target.adapters.mmrotate import MMRotateAdapter


def get_adapter(name: str):
    normalized = name.strip().lower()
    if normalized == "mmrotate":
        return MMRotateAdapter()
    raise KeyError(f"Unknown adapter '{name}'. Known adapters: mmrotate")
