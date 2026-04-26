#!/usr/bin/env python3
"""
Utility script for SLPolicyValueNetwork checkpoints.

Features:
- create: create a fresh checkpoint with randomly initialized weights
- validate: verify that one checkpoint is compatible with SLPolicyValueNetwork
- validate-all: scan a directory for .pth files and validate all
- inspect: print checkpoint keys and tensor counts

Examples:
  python save_checkpoint.py create --out model_files/sl_policy_value_bootstrap.pth
  python save_checkpoint.py validate --path model_files/sl_policy_value_bootstrap.pth
  python save_checkpoint.py validate-all --root .
  python save_checkpoint.py inspect --path model_files/sl_policy_value_bootstrap.pth
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import torch

from model_files.SLPolicyValueGPU import SLPolicyValueNetwork


def _load_state_dict_from_checkpoint(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        return checkpoint_obj["model"]
    return checkpoint_obj


def create_checkpoint(out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    model = SLPolicyValueNetwork()
    checkpoint = {
        "model": model.state_dict(),
        "meta": {
            "arch": "SLPolicyValueNetwork",
            "created_by": "save_checkpoint.py",
        },
    }
    torch.save(checkpoint, out_path)
    print(f"Created checkpoint: {out_path}")


def validate_checkpoint(path: str, strict: bool = True) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return False, f"File not found: {path}"

    try:
        obj = torch.load(path, map_location="cpu")
        state_dict = _load_state_dict_from_checkpoint(obj)

        model = SLPolicyValueNetwork()
        model.load_state_dict(state_dict, strict=strict)
        return True, "PASS"
    except Exception as e:
        return False, f"FAIL: {e}"


def validate_all(root: str, strict: bool = True) -> Dict[str, str]:
    results = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".pth"):
                full_path = os.path.join(dirpath, filename)
                ok, message = validate_checkpoint(full_path, strict=strict)
                results[full_path] = message if ok else message
    return results


def inspect_checkpoint(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    obj = torch.load(path, map_location="cpu")
    print(f"Checkpoint file: {path}")
    print(f"Top-level type: {type(obj).__name__}")

    if isinstance(obj, dict):
        print(f"Top-level keys: {list(obj.keys())}")

    state_dict = _load_state_dict_from_checkpoint(obj)
    if not isinstance(state_dict, dict):
        print("State dict is not a dictionary; cannot inspect tensor keys.")
        return

    tensor_keys = [k for k, v in state_dict.items() if torch.is_tensor(v)]
    total_params = 0
    for key in tensor_keys:
        total_params += state_dict[key].numel()

    print(f"Tensor entries in state_dict: {len(tensor_keys)}")
    print(f"Total scalar parameters: {total_params}")

    preview = tensor_keys[:10]
    if preview:
        print("First tensor keys:")
        for key in preview:
            print(f"  - {key}: {tuple(state_dict[key].shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint helper for SLPolicyValueNetwork")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a fresh checkpoint")
    p_create.add_argument("--out", required=True, help="Output checkpoint path (.pth)")

    p_validate = sub.add_parser("validate", help="Validate one checkpoint")
    p_validate.add_argument("--path", required=True, help="Checkpoint path")
    p_validate.add_argument("--non-strict", action="store_true", help="Use strict=False")

    p_validate_all = sub.add_parser("validate-all", help="Validate all .pth files under a root")
    p_validate_all.add_argument("--root", default=".", help="Root directory to scan")
    p_validate_all.add_argument("--non-strict", action="store_true", help="Use strict=False")

    p_inspect = sub.add_parser("inspect", help="Inspect one checkpoint")
    p_inspect.add_argument("--path", required=True, help="Checkpoint path")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create":
        create_checkpoint(args.out)
        return 0

    if args.command == "validate":
        strict = not args.non_strict
        ok, message = validate_checkpoint(args.path, strict=strict)
        print(f"{args.path}: {message}")
        return 0 if ok else 1

    if args.command == "validate-all":
        strict = not args.non_strict
        results = validate_all(args.root, strict=strict)

        if not results:
            print(f"No .pth files found under: {args.root}")
            return 1

        passed = 0
        for path, status in results.items():
            print(f"{path}: {status}")
            if status == "PASS":
                passed += 1

        print(f"\nSummary: {passed}/{len(results)} passed")
        return 0 if passed > 0 else 1

    if args.command == "inspect":
        inspect_checkpoint(args.path)
        return 0

    print("Unknown command")
    return 1


if __name__ == "__main__":
    sys.exit(main())
