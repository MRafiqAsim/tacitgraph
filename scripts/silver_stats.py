"""Generate comprehensive Silver layer statistics.

Reports on all processed data: threads, emails, chunks, entities,
relationships, attachments, classifications, and skipped items.

Usage:
    python scripts/silver_stats.py --silver data/silver_llm --bronze data/bronze
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Silver layer statistics")
    parser.add_argument("--silver", required=True, help="Path to Silver layer")
    parser.add_argument("--bronze", help="Path to Bronze layer (for comparison)")
    args = parser.parse_args()

    silver = Path(args.silver)

    # ============================================================
    # SILVER CHUNKS
    # ============================================================
    email_chunks = (
        list((silver / "not_personal" / "email_chunks").glob("*.json"))
        if (silver / "not_personal" / "email_chunks").exists()
        else []
    )
    att_chunks = (
        list((silver / "not_personal" / "attachment_chunks").glob("*.json"))
        if (silver / "not_personal" / "attachment_chunks").exists()
        else []
    )
    thread_summaries = (
        list((silver / "not_personal" / "thread_summaries").glob("*.json"))
        if (silver / "not_personal" / "thread_summaries").exists()
        else []
    )
    email_summaries = (
        list((silver / "not_personal" / "email_summaries").glob("*.json"))
        if (silver / "not_personal" / "email_summaries").exists()
        else []
    )
    personal = list((silver / "personal").glob("*.json")) if (silver / "personal").exists() else []

    # Parse all email chunks
    total_entities = 0
    total_relationships = 0
    total_tokens = 0
    entity_types = Counter()
    relationship_types = Counter()
    languages = Counter()
    source_emails = set()
    thread_ids = set()
    senders = Counter()
    has_summary = 0
    has_text_english = 0
    processing_modes = Counter()

    for f in email_chunks:
        try:
            with open(f) as fh:
                d = json.load(fh)

            # Counts
            entities = d.get("kg_entities", [])
            relationships = d.get("kg_relationships", [])
            total_entities += len(entities)
            total_relationships += len(relationships)
            total_tokens += d.get("token_count", 0)

            # Entity types
            for e in entities:
                entity_types[e.get("type", "UNKNOWN")] += 1

            # Relationship types
            for r in relationships:
                relationship_types[r.get("relationship", r.get("description", "UNKNOWN"))] += 1

            # Language
            languages[d.get("language", "unknown")] += 1

            # Source emails
            source_emails.update(d.get("source_email_ids", []))
            thread_ids.add(d.get("thread_id", ""))

            # Sender
            sender = d.get("email_sender", "")
            if sender:
                senders[sender] += 1

            # Summary
            if d.get("summary", "").strip():
                has_summary += 1

            # Text english
            text_en = d.get("text_english", "")
            text_orig = d.get("text_original", "")
            if text_en and text_en != text_orig:
                has_text_english += 1

            # Processing mode
            processing_modes[d.get("processing_mode", "unknown")] += 1

        except Exception as e:
            pass

    # Parse attachment chunks
    att_entities = 0
    att_relationships = 0
    att_tokens = 0
    att_filenames = []

    for f in att_chunks:
        try:
            with open(f) as fh:
                d = json.load(fh)
            att_entities += len(d.get("kg_entities", []))
            att_relationships += len(d.get("kg_relationships", []))
            att_tokens += d.get("token_count", 0)
            fname = d.get("source_attachment_filename", "")
            if fname:
                att_filenames.append(fname)
        except Exception:
            pass

    # Parse personal emails
    personal_threads = 0
    personal_emails = 0
    for f in personal:
        try:
            with open(f) as fh:
                d = json.load(fh)
            personal_threads += 1
            personal_emails += d.get("email_count", 1)
        except Exception:
            pass

    # ============================================================
    # BRONZE COMPARISON
    # ============================================================
    bronze_emails = 0
    bronze_attachments = 0
    if args.bronze:
        bronze_path = Path(args.bronze)
        bronze_emails = len(list(bronze_path.rglob("emails/**/*.json"))) - len(
            list(bronze_path.rglob("emails/**/metadata*.json"))
        )
        bronze_attachments = len(list(bronze_path.rglob("attachments/**/*.json")))

    # Checkpoint
    checkpoint_file = silver / "checkpoint.json"
    checkpoint_count = 0
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint_count = len(json.load(f))

    # ============================================================
    # PRINT REPORT
    # ============================================================
    print("=" * 70)
    print("SILVER LAYER STATISTICS")
    print("=" * 70)

    if args.bronze:
        print("\n--- Bronze Input ---")
        print(f"  Total Bronze emails:          {bronze_emails}")
        print(f"  Total Bronze attachments:     {bronze_attachments}")

    print("\n--- Processing Overview ---")
    print(f"  Checkpoint threads:           {checkpoint_count}")
    print(f"  Threads with chunks:          {len(thread_ids)}")
    print(f"  Source emails processed:      {len(source_emails)}")
    print(f"  Personal threads (skipped):   {personal_threads} ({personal_emails} emails)")
    print(f"  Processing modes:             {dict(processing_modes)}")

    print("\n--- Chunks ---")
    print(f"  Email chunks:                 {len(email_chunks)}")
    print(f"  Attachment chunks:            {len(att_chunks)}")
    print(f"  Total chunks:                 {len(email_chunks) + len(att_chunks)}")
    print(f"  Total tokens:                 {total_tokens + att_tokens:,}")

    print("\n--- Summaries ---")
    print(f"  Thread summaries:             {len(thread_summaries)}")
    print(f"  Email summaries:              {len(email_summaries)}")
    print(f"  Chunks with summary:          {has_summary}/{len(email_chunks)}")
    print(f"  Chunks with translation:      {has_text_english}/{len(email_chunks)}")

    print("\n--- KG Entities (Email Chunks) ---")
    print(f"  Total entities:               {total_entities}")
    print(f"  Unique entity types:          {len(entity_types)}")
    for etype, count in entity_types.most_common(15):
        print(f"    {etype:25s} {count}")

    print("\n--- KG Entities (Attachment Chunks) ---")
    print(f"  Total entities:               {att_entities}")
    print(f"  Total relationships:          {att_relationships}")

    print("\n--- KG Relationships (Email Chunks) ---")
    print(f"  Total relationships:          {total_relationships}")
    print(f"  Unique relationship types:    {len(relationship_types)}")
    for rtype, count in relationship_types.most_common(15):
        print(f"    {rtype:25s} {count}")

    # Attachment languages
    att_languages = Counter()
    for f in att_chunks:
        try:
            with open(f) as fh:
                d = json.load(fh)
            att_languages[d.get("language", "unknown")] += 1
        except Exception:
            pass

    print("\n--- Languages (Email Chunks) ---")
    for lang, count in languages.most_common():
        print(f"    {lang:10s} {count} chunks")

    if att_languages:
        print("\n--- Languages (Attachment Chunks) ---")
        for lang, count in att_languages.most_common():
            print(f"    {lang:10s} {count} chunks")

    # Combined
    all_languages = Counter()
    all_languages.update(languages)
    all_languages.update(att_languages)
    print("\n--- Languages (All Chunks Combined) ---")
    total_chunks = sum(all_languages.values())
    for lang, count in all_languages.most_common():
        pct = count / total_chunks * 100 if total_chunks else 0
        print(f"    {lang:10s} {count:6d} ({pct:.1f}%)")

    print("\n--- Top Senders ---")
    for sender, count in senders.most_common(10):
        print(f"    {sender:30s} {count} chunks")

    print("\n--- Attachment Files ---")
    print(f"  Total attachment chunks:      {len(att_chunks)}")
    ext_counter = Counter(Path(f).suffix.lower() for f in att_filenames if f)
    for ext, count in ext_counter.most_common(10):
        print(f"    {ext:10s} {count}")

    print("=" * 70)


if __name__ == "__main__":
    main()
