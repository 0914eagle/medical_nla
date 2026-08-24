"""Run judging requests through an external judge, whichever backend is available.

Every judge-dependent number in this project has been on hold for one reason:
the only model on this machine is the backbone, and judging Gemma's readouts
with Gemma invites the obvious objection. This runner takes the judge from
outside -- `codex exec` as a subprocess, or the OpenAI API -- so the same
requests can be scored either way and the results are interchangeable
downstream.

**Input**: JSONL of `{"id": ..., "prompt": ...}`. **Output**: JSONL of
`{"id", "response", "judge_model", "judge_backend", "judged_at"}`.

Three properties the science needs, built in rather than remembered:

1. **The judge may not be the backbone.** `--model` is checked against a
   deny-list and the run refuses rather than producing a number that a
   reviewer will throw out. Override needs `--allow-same-family`, which also
   stamps the output so the compromise travels with the data.
2. **Resumable.** Judging thousands of chains is long enough to be
   interrupted, and re-judging is both wasteful and non-deterministic. Ids
   already in the output file are skipped.
3. **Provenance.** The model, backend and timestamp go in every row. This is
   the field the paper has been leaving blank for the 438-row scoring; a
   number that cannot say who produced it should not enter a table.

`--dry-run` prices the job first: it counts tokens and prints an estimate
without contacting anything, because "we did not know it would be that many
tokens" is not a good reason to spend someone's money.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl

# The point of an external judge is that it is not the thing being judged.
BACKBONE_MARKERS = ("gemma", "nla-gemma")

# Rough characters-per-token for English prose. Only used by --dry-run, where
# being 10% off changes nothing about the decision it informs.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def check_judge_identity(model: str, allow_same_family: bool) -> None:
    lowered = model.lower()
    if any(marker in lowered for marker in BACKBONE_MARKERS):
        if not allow_same_family:
            raise SystemExit(
                f"refusing: '{model}' looks like the backbone under study.\n"
                "  Judging a model's readouts with that same model is the "
                "objection this project already withdrew a whole experiment "
                "over.\n"
                "  Pass --allow-same-family only for a deliberate sanity check, "
                "never for a reported number."
            )
        print(f"[warn] judge '{model}' shares the backbone family -- rows will "
              f"be stamped judge_same_family=true", file=sys.stderr)


def parse_codex_banner_model(*streams: str) -> str:
    """The model codex actually used, from its own banner.

    codex picks a default (gpt-5.6-sol at the time of writing) unless --model
    says otherwise, so "whatever codex chose" is not a provenance record. The
    banner names it, and the paper needs that name.

    Both streams are searched because the banner is diagnostic output, and
    which pipe it lands on is codex's business, not ours -- the first run
    recorded "codex-default" for exactly this reason.
    """
    for stream in streams:
        for line in (stream or "").splitlines()[:20]:
            stripped = line.strip()
            if stripped.startswith("model:"):
                return stripped.split(":", 1)[1].strip()
    return ""


def run_codex(prompt: str, model: str, timeout: int,
              codex_cmd: str = "codex") -> tuple[str, str]:
    """One `codex exec` invocation, read-only. Returns (answer, model_used).

    codex is an agent, not a completion endpoint. Two consequences are handled
    here rather than downstream:

    - **Its stdout is a transcript**, not an answer: a version banner, the
      echoed prompt, then the reply, then a token count. Parsing that with a
      regex would be guesswork, so `--output-last-message` is used instead --
      codex writes only the final message to a file and the file is the answer.
      stdout is kept solely to read the banner's model name, and as a fallback
      if the file comes back empty.
    - **It keeps session state and expects a git repo.** `--ephemeral` stops
      1,747 session files from accumulating and `--skip-git-repo-check` lets
      the judge run from anywhere.

    The prompt goes on stdin (`-`), so no shell quoting can mangle a clinical
    sentence.
    """
    handle, out_file = tempfile.mkstemp(suffix=".codex.txt")
    os.close(handle)
    try:
        cmd = codex_cmd.split() + [
            "exec", "--sandbox", "read-only", "--ephemeral",
            "--skip-git-repo-check", "--output-last-message", out_file,
        ]
        if model:
            cmd += ["--model", model]
        cmd += ["-"]
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"codex exec failed ({proc.returncode}): {proc.stderr[-800:]}"
            )
        answer = Path(out_file).read_text(encoding="utf-8").strip()
        if not answer:
            raise RuntimeError(
                "codex wrote no final message; stdout tail: "
                f"{proc.stdout[-400:]!r}"
            )
        return answer, parse_codex_banner_model(proc.stdout, proc.stderr)
    finally:
        Path(out_file).unlink(missing_ok=True)


def run_openai(prompt: str, model: str, timeout: int) -> str:
    from openai import OpenAI

    client = OpenAI(timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # Judging is a measurement, so it should be as close to repeatable as
        # the endpoint allows.
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip(), model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True,
                        help='JSONL of {"id", "prompt"}.')
    parser.add_argument("--out", required=True, help="JSONL, appended and resumable.")
    parser.add_argument("--backend", choices=["codex", "openai", "dry-run"],
                        default="dry-run")
    parser.add_argument("--model", default="",
                        help="Judge model. Empty lets codex use its default.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None,
                        help="Judge only the first N unjudged requests -- use a "
                        "small value for the smoke test before the full run.")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds between calls, for rate limits.")
    parser.add_argument("--allow-same-family", action="store_true")
    parser.add_argument(
        "--codex-cmd", default="codex",
        help="How to launch codex. Use a user-local path or "
        "'npx --yes @openai/codex' where a global install needs root.",
    )
    parser.add_argument("--in-price", type=float, default=0.0,
                        help="USD per 1M input tokens, for --dry-run.")
    parser.add_argument("--out-price", type=float, default=0.0,
                        help="USD per 1M output tokens, for --dry-run.")
    parser.add_argument("--out-tokens", type=int, default=64,
                        help="Assumed output tokens per request, for --dry-run.")
    args = parser.parse_args()

    requests = [
        {"id": str(r["id"]), "prompt": str(r["prompt"])}
        for r in read_jsonl(args.requests)
        if r.get("id") is not None and r.get("prompt")
    ]
    if not requests:
        raise SystemExit(f"no usable requests in {args.requests}")

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        done = {str(r.get("id")) for r in read_jsonl(out_path)}
        print(f"resuming: {len(done):,} already judged")
    todo = [r for r in requests if r["id"] not in done]

    if args.backend == "dry-run":
        in_tok = sum(estimate_tokens(r["prompt"]) for r in todo)
        out_tok = args.out_tokens * len(todo)
        print(f"requests        {len(requests):,}  ({len(todo):,} unjudged)")
        print(f"input tokens   ~{in_tok:,}")
        print(f"output tokens  ~{out_tok:,}  (at {args.out_tokens}/request)")
        print(f"longest prompt ~{max(estimate_tokens(r['prompt']) for r in todo):,} tokens")
        if args.in_price or args.out_price:
            cost = in_tok / 1e6 * args.in_price + out_tok / 1e6 * args.out_price
            print(f"estimated cost  ${cost:,.2f}  "
                  f"(in ${args.in_price}/M, out ${args.out_price}/M)")
        else:
            print("estimated cost  pass --in-price/--out-price for a figure; "
                  "rates change, so none is hard-coded here.")
        return

    check_judge_identity(args.model, args.allow_same_family)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"judging {len(todo):,} requests via {args.backend} "
          f"(model={args.model or 'backend default'})")

    if args.backend == "codex":
        def runner(prompt: str, model: str, timeout: int) -> tuple[str, str]:
            return run_codex(prompt, model, timeout, args.codex_cmd)
    else:
        runner = run_openai
    failures = 0
    # Appended one row at a time and flushed: an interrupted run keeps every
    # answer it paid for.
    # One writer per output. Two judges appending to the same file interleave
    # their lines and, worse, each snapshots the resume set at its own start --
    # so the second re-judges everything the first has not finished yet, at
    # full price, and the output ends up with duplicate ids. Both happened.
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"another judge is writing {out_path} ({lock_path} exists).\n"
            "  Wait for it, or point --out somewhere else. If no judge is "
            f"running, the last one was killed: rm {lock_path}"
        )
    os.write(lock_fd, f"{os.getpid()}\n".encode())
    os.close(lock_fd)
    atexit.register(lambda: lock_path.unlink(missing_ok=True))

    with out_path.open("a", encoding="utf-8") as f:
        for i, req in enumerate(todo, 1):
            try:
                response, model_used = runner(req["prompt"], args.model, args.timeout)
            except Exception as exc:  # noqa: BLE001 -- one bad row must not end the run
                failures += 1
                print(f"[{i}/{len(todo)}] {req['id']}: FAILED {exc}", file=sys.stderr)
                continue
            row: dict[str, Any] = {
                "id": req["id"],
                "response": response,
                # The model codex reports, not the flag we passed: an empty
                # --model means codex chose, and the choice is the provenance.
                "judge_model": model_used or args.model or f"{args.backend}-default",
                "judge_backend": args.backend,
                "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if args.allow_same_family:
                row["judge_same_family"] = True
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] ok")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"done -> {out_path}" + (f"   ({failures:,} failed, rerun to retry)"
                                   if failures else ""))


if __name__ == "__main__":
    main()
