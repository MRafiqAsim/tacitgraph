#!/usr/bin/env python3
"""
Coverage Check: Compare Bronze emails vs Silver processed chunks.

Reports how many emails were processed, skipped as personal,
skipped as empty, or unaccounted for.

Usage:
    python run_coverage_check.py --bronze ./data/bronze --silver ./data/silver_llm
"""

import argparse
import json
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))


def main():
    parser = argparse.ArgumentParser(description="Check Bronze → Silver processing coverage")
    parser.add_argument("--bronze", required=True, help="Path to Bronze layer")
    parser.add_argument("--silver", required=True, help="Path to Silver layer")
    parser.add_argument("--show-missing", type=int, default=10, help="Number of missing emails to show (default: 10)")
    args = parser.parse_args()

    bronze = Path(args.bronze)
    silver = Path(args.silver)

    # Collect processed email IDs from Silver chunks
    processed_ids = set()
    chunk_count = 0
    for f in glob.glob(str(silver / "not_personal/email_chunks/*.json")):
        with open(f) as fh:
            d = json.load(fh)
        for eid in d.get("source_email_ids", []):
            processed_ids.add(eid)
        chunk_count += 1

    # Collect personal (skipped) email IDs
    personal_ids = set()
    for f in glob.glob(str(silver / "personal/*.json")):
        with open(f) as fh:
            d = json.load(fh)
        for email in d.get("emails", []):
            rid = email.get("record_id", "")
            if rid:
                personal_ids.add(rid)

    # Collect attachment chunk count
    att_count = len(glob.glob(str(silver / "not_personal/attachment_chunks/*.json")))

    # Scan Bronze for all emails and find missing ones
    all_bronze = []
    for f in glob.glob(str(bronze / "emails/**/*.json"), recursive=True):
        with open(f) as fh:
            d = json.load(fh)
        all_bronze.append(d)

    total_bronze = len(all_bronze)
    known_ids = processed_ids | personal_ids

    missing_empty = []
    missing_with_body = []

    for d in all_bronze:
        rid = d.get("record_id", "")
        if rid and rid not in known_ids:
            body = d.get("email_body_text", "")
            subj = d.get("email_headers", {}).get("subject", "")
            sender = d.get("email_headers", {}).get("sender", "")
            if not body or not body.strip():
                missing_empty.append({"record_id": rid, "subject": subj, "sender": sender})
            else:
                missing_with_body.append({
                    "record_id": rid,
                    "subject": subj,
                    "sender": sender,
                    "body_length": len(body),
                })

    # Report
    print("=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    print(f"  Bronze emails:          {total_bronze}")
    print(f"  Silver email chunks:    {chunk_count}")
    print(f"  Silver attachment chunks: {att_count}")
    print(f"  Unique emails processed:{len(processed_ids)}")
    print(f"  Personal (skipped):     {len(personal_ids)}")
    print(f"  Empty body (skipped):   {len(missing_empty)}")
    print(f"  Has body, not processed:{len(missing_with_body)}")
    print(f"  Total accounted:        {len(processed_ids) + len(personal_ids) + len(missing_empty) + len(missing_with_body)}")
    print()

    if missing_with_body:
        show = min(args.show_missing, len(missing_with_body))
        print(f"Missing emails with content (showing {show}/{len(missing_with_body)}):")
        for m in missing_with_body[:show]:
            print(f"  {m['record_id']} | {m['sender'][:30]} | {m['subject'][:50]} | {m['body_length']} chars")

    if missing_empty and args.show_missing > 0:
        show = min(5, len(missing_empty))
        print(f"\nEmpty body emails (showing {show}/{len(missing_empty)}):")
        for m in missing_empty[:show]:
            print(f"  {m['record_id']} | {m['sender'][:30]} | {m['subject'][:50]}")

    # Diagnose thread grouping for missing emails with content
    if missing_with_body:
        print(f"\nThread grouping diagnosis for missing emails:")
        no_conv_id = 0
        no_message_id = 0
        has_both = 0
        for d in all_bronze:
            rid = d.get("record_id", "")
            if rid not in known_ids:
                body = d.get("email_body_text", "")
                if body and body.strip():
                    h = d.get("email_headers", {})
                    conv_id = h.get("conversation_id", "")
                    msg_id = h.get("message_id", "")
                    if not conv_id and not msg_id:
                        no_conv_id += 1
                    elif not conv_id:
                        no_message_id += 1
                    else:
                        has_both += 1
        print(f"  No conversation_id AND no message_id: {no_conv_id}")
        print(f"  Has message_id but no conversation_id: {no_message_id}")
        print(f"  Has conversation_id: {has_both}")

        # Show sample thread grouping details
        print(f"\nSample missing emails with thread details:")
        shown = 0
        for d in all_bronze:
            rid = d.get("record_id", "")
            if rid not in known_ids:
                body = d.get("email_body_text", "")
                if body and body.strip() and shown < 5:
                    h = d.get("email_headers", {})
                    print(f"  record_id:       {rid}")
                    print(f"  subject:         {(h.get('subject') or '')[:60]}")
                    print(f"  sender:          {h.get('sender') or ''}")
                    print(f"  conversation_id: {(h.get('conversation_id') or 'NONE')[:60]}")
                    print(f"  message_id:      {(h.get('message_id') or 'NONE')[:60]}")
                    print(f"  in_reply_to:     {(h.get('in_reply_to') or 'NONE')[:60]}")
                    print(f"  body_length:     {len(body)}")
                    print()
                    shown += 1

    print("=" * 60)


if __name__ == "__main__":
    main()
