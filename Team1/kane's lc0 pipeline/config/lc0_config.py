"""Configuration validation for the Lc0 analysis pipeline."""

from __future__ import annotations


def _positive_int(name, value, allow_zero=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        operator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {operator}, got {value}")
    return value


default_config = {
    "movetime_ms": 1000,
    "nodes": None,
    "depth": None,
    "multipv": 5,
    "top_intuitive": 3,
    "bad_threshold_cp": 80,
    "verbose_move_stats": True,
    "include_raw_info": False,
}


def load_config_from_args(args):
    """Build and validate search settings from an argparse namespace."""
    config = dict(default_config)
    movetime = getattr(args, "movetime_ms", None)
    nodes = getattr(args, "nodes", None)
    depth = getattr(args, "depth", None)

    if movetime is not None:
        config["movetime_ms"] = _positive_int("movetime_ms", movetime)
    if nodes is not None:
        config["nodes"] = _positive_int("nodes", nodes)
    if depth is not None:
        config["depth"] = _positive_int("depth", depth)
    if movetime is None and (nodes is not None or depth is not None):
        config["movetime_ms"] = None

    for name in ("multipv", "top_intuitive"):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = _positive_int(name, value)
    threshold = getattr(args, "bad_threshold_cp", None)
    if threshold is not None:
        config["bad_threshold_cp"] = _positive_int(
            "bad_threshold_cp", threshold, allow_zero=True
        )

    if hasattr(args, "verbose_move_stats"):
        config["verbose_move_stats"] = bool(args.verbose_move_stats)
    if hasattr(args, "include_raw_info"):
        config["include_raw_info"] = bool(args.include_raw_info)
    if not any(config[name] is not None for name in ("movetime_ms", "nodes", "depth")):
        config["movetime_ms"] = default_config["movetime_ms"]
    return config
