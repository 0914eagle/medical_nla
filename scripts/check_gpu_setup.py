"""Report whether the configured model would actually land on GPU.

`device_map="auto"` falls back to CPU without raising when no GPU is
visible, so a run can look healthy while being orders of magnitude too slow.
This checks the conditions that decide placement — driver visibility, device
count, free memory per device, and the configured device_map — before a long
extraction is started.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--require-free-gb",
        type=float,
        help=(
            "Exit non-zero unless this much is free on every visible GPU. For "
            "a queue to check before it starts, rather than after it has "
            "reported every run as failed."
        ),
    )
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"CUDA_VISIBLE_DEVICES = {visible}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print(
            "\nNo GPU visible to torch. device_map='auto' would place the model on "
            "CPU. Check the driver, the CUDA build of torch, and CUDA_VISIBLE_DEVICES."
        )
        raise SystemExit(1 if args.require_free_gb else 0)

    total_free = 0.0
    short = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        name = torch.cuda.get_device_name(index)
        total_free += free / 1e9
        print(
            f"  [{index}] {name}: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total"
        )
        if args.require_free_gb and free / 1e9 < args.require_free_gb:
            short.append((index, free / 1e9))

    if short:
        # An occupied card is the one failure a queue cannot survive quietly:
        # every run dies in the same nine seconds, the loop keeps going, and
        # the morning's summary table is eighteen rows of "(did not finish)".
        for index, free in short:
            print(
                f"\n[!] GPU {index} has {free:.1f} GB free, "
                f"{args.require_free_gb:.1f} GB required."
            )
        print("Another process is most likely still holding it:")
        print("  nvidia-smi --query-compute-apps=pid,used_memory --format=csv")
        raise SystemExit(1)

    for section in ("source_model", "nla_model"):
        model_cfg = cfg.get(section) or {}
        if model_cfg.get("model_id"):
            print(
                f"\n{section}: {model_cfg['model_id']}"
                f"\n  dtype={model_cfg.get('dtype')} device_map={model_cfg.get('device_map')}"
            )

    # A 12B model in bf16 needs roughly 24 GB of weights plus activation headroom.
    print(f"\nTotal free across visible GPUs: {total_free:.1f} GB")
    if total_free < 28:
        print(
            "This is tight for a 12B bf16 model (~24 GB of weights plus headroom). "
            "Expect CPU offload or OOM; free memory or expose another device."
        )
    else:
        print("Enough for a 12B bf16 model to be held on GPU.")


if __name__ == "__main__":
    main()
