"""Validate frozen decisions and the baseline-only DiReCT locked recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_decision(
    payload: dict[str, Any], *, decision_id: str, method_fragment: str
) -> None:
    require_equal(payload.get("schema_version"), 1, f"{decision_id} schema")
    require_equal(payload.get("decision_id"), decision_id, f"{decision_id} ID")
    require_equal(payload.get("decision_status"), "frozen", f"{decision_id} status")
    require_equal(payload.get("human_approved"), True, f"{decision_id} approval")
    require_equal(payload.get("result"), "FAIL", f"{decision_id} result")
    require_equal(payload.get("locked_test_read"), False, f"{decision_id} locked read")
    require_equal(payload.get("training", {}).get("final_step"), 1552, f"{decision_id} step")
    require_equal(payload.get("training", {}).get("seeds"), [17, 29, 43], f"{decision_id} seeds")
    if method_fragment not in str(payload.get("method") or ""):
        raise ValueError(f"{decision_id} method does not contain {method_fragment!r}")
    gate = payload.get("frozen_gate")
    if not isinstance(gate, dict) or not gate:
        raise ValueError(f"{decision_id} frozen gate is missing")
    if any(value is not False for key, value in gate.items() if key != "retained_original_nll_noninferior_each_seed"):
        raise ValueError(f"{decision_id} contains an unexpected passing primary gate")


def validate_recipe(
    recipe: dict[str, Any], d19_path: Path, d21_path: Path
) -> None:
    require_equal(recipe.get("schema_version"), 1, "recipe schema")
    require_equal(recipe.get("recipe_id"), "direct_locked_baseline_only_v1", "recipe ID")
    require_equal(recipe.get("human_approved"), True, "recipe approval")
    require_equal(
        recipe.get("medical_nla_locked_generation_authorized"),
        False,
        "Medical-NLA locked authorization",
    )
    access = recipe.get("locked_access") or {}
    require_equal(
        access.get("table_1a", {}).get("populations"),
        {"test_seen": 72, "test_pdd_heldout": 106},
        "Table 1A populations",
    )
    require_equal(
        access.get("table_1b", {}).get("populations"),
        {"test_seen": 72},
        "Table 1B populations",
    )
    require_equal(access.get("table_1b", {}).get("layer"), 24, "Table 1B layer")
    require_equal(
        access.get("table_2", {}).get("methods"), ["cot", "vanilla"], "Table 2 methods"
    )
    require_equal(
        access.get("table_2", {}).get("populations"),
        {"test_seen": 72, "test_pdd_heldout": 106},
        "Table 2 populations",
    )
    generation = access.get("table_2", {}).get("generation") or {}
    require_equal(generation.get("method"), "vanilla", "Table 2 generated method")
    require_equal(generation.get("layer"), 32, "Table 2 layer")
    require_equal(generation.get("do_sample"), False, "Table 2 sampling")
    require_equal(generation.get("max_new_tokens"), 512, "Table 2 max tokens")

    dependencies = {
        str(item.get("decision_id")): item
        for item in recipe.get("decision_dependencies") or []
    }
    for decision_id, path in (("D19", d19_path), ("D21", d21_path)):
        dependency = dependencies.get(decision_id)
        if dependency is None:
            raise ValueError(f"Recipe misses {decision_id} dependency")
        require_equal(dependency.get("sha256"), sha256_file(path), f"{decision_id} recipe hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d19", required=True, type=Path)
    parser.add_argument("--d21", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    args = parser.parse_args()

    d19 = load_json(args.d19)
    d21 = load_json(args.d21)
    recipe = load_json(args.recipe)
    validate_decision(d19, decision_id="D19", method_fragment="D10")
    validate_decision(d21, decision_id="D21", method_fragment="D20")
    validate_recipe(recipe, args.d19, args.d21)
    print(
        "[locked recipe] validated D19=FAIL D21=FAIL; "
        "authorized methods=cot,vanilla; Medical-NLA locked generation=false"
    )


if __name__ == "__main__":
    main()
