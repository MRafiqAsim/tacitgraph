#!/usr/bin/env python3
"""
Show processing coverage — which years are represented in Silver chunks
and how many threads are in the checkpoint.

Usage:
    python show_coverage.py --silver ./data/silver_llm
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Show Silver layer processing coverage")
    parser.add_argument(
        "--silver",
        "-s",
        type=str,
        required=True,
        help="Path to Silver layer directory (e.g. ./data/silver_llm)",
    )
    args = parser.parse_args()

    silver = Path(args.silver)
    chunk_dir = silver / "not_personal" / "email_chunks"
    checkpoint_file = silver / "checkpoint.json"

    # Checkpoint stats
    if checkpoint_file.exists():
        ids = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        print(f"Checkpoint: {len(ids)} threads processed\n")
    else:
        print("No checkpoint.json found\n")

    # Year coverage from chunks
    if not chunk_dir.exists():
        print(f"ERROR: {chunk_dir} does not exist.")
        return

    files = list(chunk_dir.glob("*.json"))
    print(f"Total chunks in Silver: {len(files)}\n")

    years = Counter()
    errors = 0

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            date = data.get("sent_timestamp") or data.get("email_date") or data.get("date", "")
            if date:
                year = str(date)[:4]
                if year.isdigit() and 1990 <= int(year) <= 2030:
                    years[year] += 1
        except Exception:
            errors += 1

    print("Year coverage (chunks per year):")
    print("-" * 30)
    for year in sorted(years):
        print(f"  {year}: {years[year]:>6} chunks")
    print("-" * 30)
    print(f"  Total: {sum(years.values()):>6} chunks")

    if errors:
        print(f"\nWarning: {errors} files could not be read")


if __name__ == "__main__":
    main()
