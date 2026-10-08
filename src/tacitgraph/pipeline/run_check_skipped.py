#!/usr/bin/env python3
"""
Analyze skipped emails — shows why emails were skipped and checks
if attachments were missed.

Usage:
    python run_check_skipped.py --bronze ./data/bronze --silver ./data/silver_llm
    python run_check_skipped.py --bronze ./data/bronze --silver ./data/silver_llm --show 20
"""

import argparse
import glob
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Analyze skipped emails")
    parser.add_argument("--bronze", required=True, help="Path to Bronze layer")
    parser.add_argument("--silver", required=True, help="Path to Silver layer")
    parser.add_argument(
        "--show", type=int, default=10, help="Number of skipped emails to show (default: 10)"
    )
    args = parser.parse_args()

    # Collect processed email IDs
    known = set()
    for f in glob.glob(f"{args.silver}/not_personal/email_chunks/*.json"):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        known.update(d.get("source_email_ids", []))
    for f in glob.glob(f"{args.silver}/personal/*.json"):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for e in d.get("emails", []):
            known.add(e.get("record_id", ""))

    # Analyze skipped emails
    empty_no_att = []
    empty_with_att = []
    has_body_skipped = []

    for f in glob.glob(f"{args.bronze}/emails/**/*.json", recursive=True):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        rid = d.get("record_id", "")
        if rid in known:
            continue

        body = d.get("email_body_text", "")
        meta = d.get("document_metadata", {})
        h = d.get("email_headers", {})

        has_att_flag = meta.get("has_attachments", False)
        att_count = meta.get("attachment_count", 0)

        # Check bronze attachments folder for actual attachments
        att_folder = Path(args.bronze) / "attachments" / rid
        actual_att_files = list(att_folder.glob("*")) if att_folder.exists() else []

        info = {
            "record_id": rid,
            "subject": (h.get("subject") or "")[:60],
            "sender": (h.get("sender") or "")[:30],
            "body_length": len(body),
            "has_attachments_flag": has_att_flag,
            "attachment_count_meta": att_count,
            "actual_att_files": len(actual_att_files),
            "att_folder_exists": att_folder.exists(),
        }

        if not body or not body.strip():
            if has_att_flag or actual_att_files:
                empty_with_att.append(info)
            else:
                empty_no_att.append(info)
        else:
            has_body_skipped.append(info)

    # Report
    print("=" * 70)
    print("SKIPPED EMAILS ANALYSIS")
    print("=" * 70)
    print(f"  Total processed: {len(known)}")
    print(f"  Empty body, no attachments: {len(empty_no_att)}")
    print(f"  Empty body, HAS attachments: {len(empty_with_att)}")
    print(f"  Has body, not processed: {len(has_body_skipped)}")

    if empty_with_att:
        print(f"\n--- Empty body WITH attachments ({len(empty_with_att)}) ---")
        print("These emails have content in attachments but were skipped:")
        for info in empty_with_att[: args.show]:
            print(f"  {info['record_id']}")
            print(f"    subject: {info['subject']}")
            print(f"    sender: {info['sender']}")
            print(f"    has_attachments flag: {info['has_attachments_flag']}")
            print(f"    attachment_count meta: {info['attachment_count_meta']}")
            print(f"    actual att files on disk: {info['actual_att_files']}")
            print(f"    att folder exists: {info['att_folder_exists']}")
            print()

    if has_body_skipped:
        print(f"\n--- Has body but not processed ({len(has_body_skipped)}) ---")
        for info in has_body_skipped[: args.show]:
            print(
                f"  {info['record_id']} | {info['sender']} | {info['subject']} | {info['body_length']} chars"
            )

    if empty_no_att:
        print(f"\n--- Empty body, no attachments ({len(empty_no_att)}) ---")
        for info in empty_no_att[: min(5, args.show)]:
            print(f"  {info['record_id']} | {info['sender']} | {info['subject']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
