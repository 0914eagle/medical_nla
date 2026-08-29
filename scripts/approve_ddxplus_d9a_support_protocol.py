"""Record explicit human approval of an unchanged D9a cut recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.make_ddxplus_d9a_supported_pairs import sha256_file


CONFIRMATION = "I_APPROVE_D9A_CUTS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation", required=True, type=Path)
    parser.add_argument("--validation-scores", required=True, type=Path)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.confirmation != CONFIRMATION:
        raise ValueError(f"--confirmation must equal {CONFIRMATION}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite approval: {args.output}")
    payload = json.loads(args.recommendation.read_text(encoding="utf-8"))
    if payload.get("human_approved") is not False:
        raise ValueError("Recommendation must be explicitly unapproved")
    observed_hash = sha256_file(args.validation_scores)
    if payload.get("validation_scores_sha256") != observed_hash:
        raise ValueError("Recommendation does not match validation score SHA256")
    selected = payload.get("selected") or {}
    if not selected.get("meets_false_support_cap"):
        raise ValueError("Recommended cuts do not satisfy the frozen false-support cap")
    if float(selected.get("false_support_rate", 1.0)) > 0.05:
        raise ValueError("Recommended cuts exceed false-support rate 0.05")
    payload.update(
        {
            "human_approved": True,
            "approved_by": args.approved_by.strip(),
            "approved_at": args.approved_at.strip(),
            "recommendation": str(args.recommendation),
            "recommendation_sha256": sha256_file(args.recommendation),
        }
    )
    if not payload["approved_by"] or not payload["approved_at"]:
        raise ValueError("Approval identity and timestamp cannot be empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[approved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
