"""Generate comprehensive Bronze layer statistics.

Reports on all ingested data: emails, threads, attachments, senders,
recipients, languages, date ranges, and data quality.

Usage:
    python scripts/bronze_stats.py --bronze data/bronze
"""

import argparse
import json
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Bronze layer statistics")
    parser.add_argument("--bronze", required=True, help="Path to Bronze layer")
    args = parser.parse_args()

    bronze = Path(args.bronze)
    emails_dir = bronze / "emails"
    attachments_dir = bronze / "attachments"

    # ============================================================
    # EMAILS
    # ============================================================
    total_emails = 0
    senders = Counter()
    sender_emails_counter = Counter()
    recipients_count = 0
    unique_recipients = set()
    languages = Counter()
    years = Counter()
    has_body = 0
    empty_body = 0
    has_attachments = 0
    total_attachment_count = 0
    has_message_id = 0
    has_in_reply_to = 0
    has_references = 0
    body_lengths = []
    subjects = []

    for f in emails_dir.rglob("*.json"):
        if "metadata" in f.name:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)

            total_emails += 1
            headers = d.get("email_headers", {})
            meta = d.get("document_metadata", {})

            # Sender
            sender = headers.get("sender", "unknown")
            sender_email = headers.get("sender_email", "")
            senders[sender] += 1
            if sender_email:
                sender_emails_counter[sender_email] += 1

            # Recipients
            for field in ["recipients_to", "recipients_cc", "recipients_bcc"]:
                recips = headers.get(field, [])
                if isinstance(recips, list):
                    recipients_count += len(recips)
                    for r in recips:
                        if isinstance(r, dict):
                            unique_recipients.add(r.get("email", ""))
                        elif isinstance(r, str):
                            unique_recipients.add(r)

            # Language
            lang = meta.get("language", None) or "unknown"
            languages[lang] += 1

            # Date
            sent_time = meta.get("sent_time", "")
            if sent_time:
                try:
                    year = str(sent_time)[:4]
                    if year.isdigit():
                        years[year] += 1
                except Exception:
                    pass

            # Body
            body = d.get("email_body_text", "") or ""
            if body.strip():
                has_body += 1
                body_lengths.append(len(body))
            else:
                empty_body += 1

            # Attachments
            att_flag = meta.get("has_attachments", False)
            att_count = meta.get("attachment_count", 0)
            if att_flag or att_count > 0:
                has_attachments += 1
                total_attachment_count += att_count

            # Headers
            if headers.get("message_id"):
                has_message_id += 1
            if headers.get("in_reply_to"):
                has_in_reply_to += 1
            refs = headers.get("references", [])
            if refs and len(refs) > 0:
                has_references += 1

            # Subject
            subj = headers.get("subject", "")
            if subj:
                subjects.append(subj)

        except Exception:
            pass

    # ============================================================
    # ATTACHMENTS
    # ============================================================
    total_att_files = 0
    att_extensions = Counter()
    att_with_text = 0
    att_failed = 0

    if attachments_dir.exists():
        for att_dir in attachments_dir.iterdir():
            if not att_dir.is_dir():
                continue
            for f in att_dir.iterdir():
                if f.suffix == ".json":
                    try:
                        with open(f) as fh:
                            d = json.load(fh)
                        total_att_files += 1
                        ext = Path(d.get("filename", "")).suffix.lower() or ".unknown"
                        att_extensions[ext] += 1
                        if d.get("extraction_success", False):
                            att_with_text += 1
                        else:
                            att_failed += 1
                    except Exception:
                        pass

    # ============================================================
    # THREAD ESTIMATE
    # ============================================================
    normalized_subjects = Counter()
    for subj in subjects:
        clean = subj.strip()
        for prefix in ["RE:", "Re:", "re:", "FW:", "Fw:", "fw:", "AW:", "Aw:", "FINAL "]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):].strip()
        normalized_subjects[clean] += 1

    single_email_threads = sum(1 for c in normalized_subjects.values() if c == 1)
    multi_email_threads = sum(1 for c in normalized_subjects.values() if c > 1)

    # ============================================================
    # PRINT REPORT
    # ============================================================
    print("=" * 70)
    print("BRONZE LAYER STATISTICS")
    print("=" * 70)

    print(f"\n--- Emails ---")
    print(f"  Total emails:                 {total_emails}")
    print(f"  With body text:               {has_body}")
    print(f"  Empty body:                   {empty_body}")
    print(f"  With attachments:             {has_attachments} ({total_attachment_count} total attachments)")

    if body_lengths:
        print(f"\n--- Body Size ---")
        print(f"  Average body length:          {sum(body_lengths) // len(body_lengths):,} chars")
        print(f"  Min body length:              {min(body_lengths):,} chars")
        print(f"  Max body length:              {max(body_lengths):,} chars")
        print(f"  Emails > 10K chars:           {sum(1 for l in body_lengths if l > 10000)}")
        print(f"  Emails > 20K chars:           {sum(1 for l in body_lengths if l > 20000)}")

    print(f"\n--- Threading Headers ---")
    print(f"  Has Message-ID:               {has_message_id}/{total_emails}")
    print(f"  Has In-Reply-To:              {has_in_reply_to}/{total_emails}")
    print(f"  Has References:               {has_references}/{total_emails}")

    print(f"\n--- Thread Estimate (by subject) ---")
    print(f"  Unique subjects:              {len(normalized_subjects)}")
    print(f"  Single-email threads:         {single_email_threads}")
    print(f"  Multi-email threads:          {multi_email_threads}")

    print(f"\n--- Participants ---")
    print(f"  Unique senders:               {len(senders)}")
    print(f"  Unique sender emails:         {len(sender_emails_counter)}")
    print(f"  Total recipients:             {recipients_count}")
    print(f"  Unique recipient emails:      {len(unique_recipients)}")

    print(f"\n--- Top Senders ---")
    for sender, count in senders.most_common(15):
        print(f"    {sender:35s} {count}")

    print(f"\n--- Languages ---")
    for lang, count in languages.most_common():
        pct = count / total_emails * 100 if total_emails else 0
        print(f"    {lang:10s} {count:6d} ({pct:.1f}%)")

    print(f"\n--- Date Range ---")
    for year, count in sorted(years.items()):
        bar = "#" * (count // 20)
        print(f"    {year}:  {count:5d}  {bar}")

    print(f"\n--- Attachments on Disk ---")
    print(f"  Total attachment metadata:    {total_att_files}")
    print(f"  Successfully extracted:       {att_with_text}")
    print(f"  Failed extraction:            {att_failed}")

    if att_extensions:
        print(f"\n--- Attachment Types ---")
        for ext, count in att_extensions.most_common(15):
            print(f"    {ext:15s} {count}")

    print("=" * 70)


if __name__ == "__main__":
    main()
