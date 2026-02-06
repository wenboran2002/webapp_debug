import os
from typing import Any, Dict, Optional


_DEFAULT_OPTIMIZER_CONFIG: Dict[str, Any] = {
    "optimize": {
        "steps": 100,
        "print_every": 10,
        "optimize_interval": 3,
    },
    "loss_weights": {
        "contact": 50.0,
        "collision": 8.0,
        "mask": 0.05,
    },
    "smoothing": {
        "alpha": 0.25,
        "beta": 0.25,
        "window": 7,
        "passes": 2,
        "method": "ema_box",
    },
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _default_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "configs", "optimizer.yaml")


def load_optimizer_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load optimizer config from YAML.

    - If `config_path` is None, loads `video_optimizer/configs/optimizer.yaml`.
    - If PyYAML isn't available or file missing, returns built-in defaults.
    """

    path = config_path or _default_config_path()
    cfg: Dict[str, Any] = {
        "optimize": dict(_DEFAULT_OPTIMIZER_CONFIG["optimize"]),
        "loss_weights": dict(_DEFAULT_OPTIMIZER_CONFIG["loss_weights"]),
        "smoothing": dict(_DEFAULT_OPTIMIZER_CONFIG["smoothing"]),
    }

    if not os.path.exists(path):
        return cfg

    try:
        import yaml  # type: ignore
    except Exception:
        # Keep pipeline runnable even without extra deps.
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    except Exception:
        return cfg

    if isinstance(user_cfg, dict):
        _deep_update(cfg, user_cfg)
    return cfg
