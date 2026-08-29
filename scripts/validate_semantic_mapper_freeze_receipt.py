"""Validate the AI-only mapper gate receipt before locked semantic scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("all_gates_passed") is not True:
        raise ValueError("Semantic mapper G1-G4 have not all passed")
    if receipt.get("locked_test_read") is not False:
        raise ValueError("Mapper receipt was not created validation-only")
    if receipt.get("protocol_sha256") != args.expected_protocol_sha256:
        raise ValueError("Semantic protocol hash mismatch in mapper receipt")
    primary = str(receipt.get("primary_model_id") or "").strip()
    auditor = str(receipt.get("auditor_model_id") or "").strip()
    if not primary or not auditor or primary == auditor:
        raise ValueError("Primary and auditor model IDs must be distinct and non-empty")
    for model_id in (primary, auditor):
        if "gemma" in model_id.casefold():
            raise ValueError("Gemma-family models cannot serve as mapper or auditor")
    gates = receipt.get("gates") or {}
    if not all((gates.get(name) or {}).get("passed") is True for name in ("G1", "G2", "G3", "G4")):
        raise ValueError("Receipt lacks an explicit pass for every mapper gate")
    for key in ("alias_table", "mapper_prompt", "ontology", "scorer"):
        artifact = receipt.get(key) or {}
        path = Path(str(artifact.get("path") or ""))
        expected = str(artifact.get("sha256") or "")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Mapper artifact mismatch: {key}")
    print(
        f"[mapper receipt] G1-G4 passed primary={primary} auditor={auditor}",
        flush=True,
    )


if __name__ == "__main__":
    main()
