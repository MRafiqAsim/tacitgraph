#!/usr/bin/env python3
"""
Build a draft entity catalog from Silver layer chunks.

Scans all Silver chunk JSONs, collects unique entities and their
LLM-discovered aliases, and saves a draft catalog for manual review
before running Gold indexing.

Usage:
    python run_build_catalog.py --silver ./data/silver_local --output ./data/gold_local/entity_catalog.json
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

# Add project paths

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_catalog(silver_path: str) -> dict:
    """Scan Silver chunks and build entity catalog."""
    silver = Path(silver_path)
    chunk_files = []
    for pattern in ["not_personal/email_chunks/*.json", "not_personal/attachment_chunks/*.json"]:
        chunk_files.extend(silver.glob(pattern))

    logger.info(f"Scanning {len(chunk_files)} chunk files...")

    # Collect entities: name -> {type, aliases, mention_count, chunks}
    entities = defaultdict(
        lambda: {
            "type": "UNKNOWN",
            "aliases": set(),
            "mention_count": 0,
            "source_chunks": [],
        }
    )

    for chunk_file in chunk_files:
        try:
            with open(chunk_file, encoding="utf-8") as f:
                chunk_data = json.load(f)

            chunk_id = chunk_data.get("chunk_id", chunk_file.stem)

            for entity in chunk_data.get("kg_entities", []):
                name = (entity.get("entity") or entity.get("text", "")).strip()
                if not name or len(name) < 2:
                    continue

                etype = entity.get("type", "UNKNOWN").upper()
                aliases = entity.get("aliases", [])

                entry = entities[name]
                entry["type"] = etype
                entry["mention_count"] += 1
                entry["source_chunks"].append(chunk_id)
                for alias in aliases:
                    if alias and alias != name:
                        entry["aliases"].add(alias)

        except Exception as e:
            logger.warning(f"Error reading {chunk_file.name}: {e}")

    # Convert sets to sorted lists and limit source_chunks
    catalog = {"entities": {}, "metadata": {}}
    for name, info in sorted(entities.items(), key=lambda x: -x[1]["mention_count"]):
        catalog["entities"][name] = {
            "standard_name": name,
            "type": info["type"],
            "aliases": sorted(info["aliases"]),
            "mention_count": info["mention_count"],
        }

    catalog["metadata"] = {
        "total_entities": len(catalog["entities"]),
        "total_aliases": sum(len(e["aliases"]) for e in catalog["entities"].values()),
        "source": "draft_from_silver",
    }

    return catalog


def main():
    parser = argparse.ArgumentParser(description="Build draft entity catalog from Silver chunks")
    parser.add_argument("--silver", required=True, help="Path to Silver layer")
    parser.add_argument("--output", required=True, help="Output path for catalog JSON")
    args = parser.parse_args()

    catalog = build_catalog(args.silver)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(
        f"\nCatalog built: {catalog['metadata']['total_entities']} entities, "
        f"{catalog['metadata']['total_aliases']} aliases"
    )
    print(f"Saved to: {args.output}")
    print("\nReview the catalog and merge/fix aliases before running Gold indexing.")


if __name__ == "__main__":
    main()
