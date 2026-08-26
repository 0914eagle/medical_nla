"""Run generic JSONL judge requests through the local native Llama checkpoint.

This keeps restricted DiReCT readouts on the research server. Launch with
``torchrun --nproc_per_node 1``; the official native Llama implementation
initializes model parallel even for a single visible GPU.
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.jsonl import read_jsonl


def load_llama_class(official_repo: Path) -> Any:
    repo = str(official_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return importlib.import_module("llama").Llama


def completion_batch(
    generator: Any,
    prompts: list[str],
    *,
    max_gen_len: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    dialogs = [[{"role": "user", "content": prompt}] for prompt in prompts]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return [str(result["generation"]["content"]).strip() for result in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--ckpt-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-gen-len", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge-model", default="Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    for path in (args.requests, args.official_repo, args.ckpt_dir, args.tokenizer_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_batch_size <= 0 or args.max_gen_len <= 0:
        raise ValueError("Batch and generation lengths must be positive")

    requests = list(read_jsonl(args.requests))
    request_ids = [str(row["id"]) for row in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Duplicate request IDs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {str(row["id"]) for row in read_jsonl(args.out)} if args.out.exists() else set()
    todo = [row for row in requests if str(row["id"]) not in done]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        todo = todo[: args.limit]
    print(f"[judge] requests={len(requests)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    lock_path = args.out.with_suffix(args.out.suffix + ".lock")
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"Another writer holds {lock_path}") from exc
    os.write(lock_fd, f"{os.getpid()}\n".encode())
    os.close(lock_fd)
    atexit.register(lambda: lock_path.unlink(missing_ok=True))

    llama_class = load_llama_class(args.official_repo)
    generator = llama_class.build(
        ckpt_dir=str(args.ckpt_dir),
        tokenizer_path=str(args.tokenizer_path),
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    with args.out.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), args.max_batch_size):
            batch = todo[start : start + args.max_batch_size]
            responses = completion_batch(
                generator,
                [str(row["prompt"]) for row in batch],
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            if len(responses) != len(batch):
                raise RuntimeError("Judge returned a different number of completions")
            for request, response in zip(batch, responses):
                record = {
                    "id": str(request["id"]),
                    "response": response,
                    "judge_model": args.judge_model,
                    "judge_backend": "direct-native-llama",
                    "judged_at": timestamp,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed = min(start + len(batch), len(todo))
            print(f"[judge] {completed}/{len(todo)}", flush=True)

    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
