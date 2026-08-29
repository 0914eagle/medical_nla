"""Dump the exact Vanilla AV actor prompt and resolved checkpoint metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.modeling import load_tokenizer
from src.nla import load_nla_sidecar


def snapshot_revision(path: str) -> str:
    parts = Path(path).parts
    if "snapshots" not in parts:
        raise ValueError(f"Cannot resolve Hugging Face snapshot revision from {path}")
    index = parts.index("snapshots")
    if index + 1 >= len(parts) or not parts[index + 1]:
        raise ValueError(f"Missing snapshot revision in {path}")
    return parts[index + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prompt-output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["nla_model"]
    model_id = str(model_cfg["model_id"])
    tokenizer = load_tokenizer(
        model_id,
        cache_dir=cfg["paths"].get("cache_dir"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    sidecar = load_nla_sidecar(
        model_id,
        tokenizer=tokenizer,
        cache_dir=cfg["paths"].get("cache_dir"),
        filename=model_cfg.get("sidecar_filename", "nla_meta.yaml"),
        expected_d_model=model_cfg.get("expected_d_model"),
        expected_injection_token_id=model_cfg.get("expected_injection_token_id"),
    )
    prompt = sidecar.actor_prompt_template
    if not prompt.endswith("\n"):
        prompt += "\n"
    args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_output.write_text(prompt, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "model_id": model_id,
        "snapshot_revision": snapshot_revision(sidecar.path),
        "sidecar_path": sidecar.path,
        "sidecar_filename": model_cfg.get("sidecar_filename", "nla_meta.yaml"),
        "d_model": sidecar.d_model,
        "injection_token_id": sidecar.injection_token_id,
        "injection_scale": sidecar.injection_scale,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[actor prompt] model={model_id} revision={metadata['snapshot_revision']} "
        f"prompt={args.prompt_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
