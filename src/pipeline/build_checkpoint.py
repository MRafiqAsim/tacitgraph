#!/usr/bin/env python3
"""
Build checkpoint.json from existing Silver layer chunks.

Run this once to create a checkpoint from already-processed Silver files,
so that --resume in run_thread_processing.py skips them correctly.

Usage:
    python build_checkpoint.py --silver ./data/silver_llm
    python build_checkpoint.py --silver ./data/silver_local
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build checkpoint from existing Silver chunks")
    parser.add_argument(
        "--silver", "-s",
        type=str,
        required=True,
        help="Path to Silver layer directory (e.g. ./data/silver_llm)"
    )
    args = parser.parse_args()

    silver = Path(args.silver)
    chunk_dir = silver / "not_personal" / "email_chunks"

    if not chunk_dir.exists():
        print(f"ERROR: {chunk_dir} does not exist.")
        return

    thread_ids = set()
    errors = 0

    files = list(chunk_dir.glob("*.json"))
    print(f"Scanning {len(files)} chunk files in {chunk_dir} ...")

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            tid = data.get("thread_id")
            if tid:
                thread_ids.add(tid)
        except Exception as e:
            errors += 1

    checkpoint_file = silver / "checkpoint.json"
    checkpoint_file.write_text(json.dumps(list(thread_ids)), encoding="utf-8")

    print(f"Checkpoint created: {checkpoint_file}")
    print(f"  Thread IDs saved: {len(thread_ids)}")
    if errors:
        print(f"  Errors reading files: {errors}")


if __name__ == "__main__":
    main()
