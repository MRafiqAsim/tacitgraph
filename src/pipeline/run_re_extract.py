#!/usr/bin/env python3
"""
Re-extract entities and relationships on existing Silver chunks.

Keeps chunking, summaries, and text unchanged. Only updates kg_entities
and kg_relationships using the latest LLM prompts and entity config.

Usage:
    python run_re_extract.py --silver ./data/silver_llm
    python run_re_extract.py --silver ./data/silver_llm --workers 4
"""

import argparse
import json
import os
import glob
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_chunk(cf, api_key, endpoint, deployment):
    """Process a single chunk — called by each thread."""
    from silver.kg_entity_extractor import LLMKGExtractor
    from silver.relationship_extractor import LLMRelationshipExtractor

    ee = LLMKGExtractor(api_key=api_key, use_azure=True, azure_endpoint=endpoint, azure_deployment=deployment)
    re_ext = LLMRelationshipExtractor(api_key=api_key, use_azure=True, azure_endpoint=endpoint, azure_deployment=deployment)

    with open(cf) as f:
        c = json.load(f)

    text = c.get("text_english") or c.get("text_anonymized", "")
    lang = c.get("language", "en")
    chunk_id = c.get("chunk_id", os.path.basename(cf))

    if not text.strip():
        return chunk_id, 0, 0, None

    # Prepend email metadata for better entity extraction (full names, facility refs)
    subject = c.get("thread_subject", "")
    sender = c.get("email_sender", "")
    recipients = c.get("email_recipients_to", [])
    prefix = ""
    if subject:
        prefix += f"Subject: {subject}\n"
    if sender:
        prefix += f"From: {sender}\n"
    if recipients:
        recip_names = [r.get("name", "") if isinstance(r, dict) else str(r) for r in recipients]
        recip_names = [n for n in recip_names if n]
        if recip_names:
            prefix += f"To: {', '.join(recip_names)}\n"
    if prefix:
        text = prefix + "\n" + text

    try:
        ents = ee.extract(text, lang)
        seen = set()
        unique = [e for e in ents if not (
            (e.entity.lower(), e.entity_type) in seen or seen.add((e.entity.lower(), e.entity_type))
        )]
        ed = [e.to_dict() for e in unique]
        rd = [r.to_dict() for r in re_ext.extract(text, unique, lang)] if len(unique) >= 2 else []

        c["kg_entities"] = ed
        c["kg_relationships"] = rd

        with open(cf, "w") as f:
            json.dump(c, f, indent=2, ensure_ascii=False, default=str)

        return chunk_id, len(ed), len(rd), None

    except Exception as e:
        return chunk_id, 0, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Re-extract entities and relationships on existing Silver chunks")
    parser.add_argument("--silver", required=True, help="Path to Silver layer")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process (default: all)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1, recommended: 3-5)")
    args = parser.parse_args()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    if not api_key or not endpoint:
        print("ERROR: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in .env")
        sys.exit(1)

    silver = Path(args.silver)
    chunks = sorted(
        glob.glob(str(silver / "not_personal/email_chunks/*.json")) +
        glob.glob(str(silver / "not_personal/attachment_chunks/*.json"))
    )

    if args.limit:
        chunks = chunks[:args.limit]

    total = len(chunks)
    print(f"Re-extracting {total} chunks from {args.silver} ({args.workers} workers)")
    print("-" * 50)

    start_time = time.time()
    total_e, total_r, errors = 0, 0, 0
    completed = 0

    if args.workers <= 1:
        # Sequential
        for i, cf in enumerate(chunks):
            chunk_id, ne, nr, err = process_chunk(cf, api_key, endpoint, deployment)
            completed += 1
            total_e += ne
            total_r += nr
            if err:
                errors += 1
                print(f"  [{completed}/{total}] ERROR {chunk_id}: {err}")
            if completed % 25 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] {total_e}e {total_r}r | {rate:.1f} chunks/sec | ETA: {eta/60:.1f}min")
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_chunk, cf, api_key, endpoint, deployment): cf
                for cf in chunks
            }
            for future in as_completed(futures):
                chunk_id, ne, nr, err = future.result()
                completed += 1
                total_e += ne
                total_r += nr
                if err:
                    errors += 1
                    print(f"  [{completed}/{total}] ERROR {chunk_id}: {err}")
                if completed % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"  [{completed}/{total}] {total_e}e {total_r}r | {rate:.1f} chunks/sec | ETA: {eta/60:.1f}min")

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"  {total_e} entities, {total_r} relationships, {errors} errors")
    print(f"  {completed/elapsed:.1f} chunks/sec")


if __name__ == "__main__":
    main()
