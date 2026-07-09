from __future__ import annotations

import argparse
from pathlib import Path

from .jsonl import read_jsonl


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# NLA Report: {Path(args.input).name}",
        "",
        "| id | prompt | query | output | layer | position |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(str(row.get("id", ""))),
                    md_escape(str(row.get("prompt", ""))),
                    md_escape(str(row.get("query", ""))),
                    md_escape(str(row.get("nla_output", ""))),
                    md_escape(str(row.get("layer", ""))),
                    md_escape(str(row.get("position", ""))),
                ]
            )
            + " |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
