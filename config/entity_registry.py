"""
Entity Registry Module

Loads config/entity_config.json once and exposes helper functions
used throughout the pipeline for consistent entity type handling.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "entity_config.json"
_config = None
_catalog = None
_catalog_path = None


def _load_config() -> dict:
    global _config
    if _config is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _config = json.load(f)
    return _config


@lru_cache(maxsize=1)
def _build_type_alias_map() -> Dict[str, str]:
    """Build mapping from alias/spacy types to standard types."""
    cfg = _load_config()
    alias_map: Dict[str, str] = {}
    for standard, info in cfg["entity_types"].items():
        for alias in info.get("llm_aliases", []):
            alias_map[alias.upper()] = standard
        for label in info.get("spacy_labels", []):
            alias_map[label.upper()] = standard
    return alias_map


# ------------------------------------------------------------------
# Entity type helpers
# ------------------------------------------------------------------

def get_standard_type(raw_type: str) -> str:
    """Normalize an entity type to its standard form.

    "ORGANIZATION" -> "ORG", "GEO" -> "GPE", unknown -> pass-through.
    """
    upper = raw_type.upper()
    cfg = _load_config()
    if upper in cfg["entity_types"]:
        return upper
    return _build_type_alias_map().get(upper, upper)


def get_all_entity_types() -> Set[str]:
    return set(_load_config()["entity_types"].keys())


def get_all_accepted_types() -> Set[str]:
    """All standard types plus all known aliases (upper-cased)."""
    cfg = _load_config()
    accepted = set(cfg["entity_types"].keys())
    for info in cfg["entity_types"].values():
        for alias in info.get("llm_aliases", []):
            accepted.add(alias.upper())
        for label in info.get("spacy_labels", []):
            accepted.add(label.upper())
    return accepted


def get_pii_types() -> Set[str]:
    return {"PERSON"}


def get_path_index_types() -> Set[str]:
    cfg = _load_config()
    return {t for t, info in cfg["entity_types"].items() if info.get("index_paths", False)}


def get_relationship_types_for_prompt() -> str:
    """Build relationship type list for LLM prompt injection.

    Returns a formatted string like:
    - WORKS_AT: person employed at organization (person → organization)
    - LOCATED_IN: entity is in a location (entity → location)
    """
    cfg = _load_config()
    semantic = cfg["relationship_types"]["semantic"]["types"]
    lines = []
    for rtype, info in semantic.items():
        desc = info["description"]
        direction = info["direction"]
        lines.append(f"- {rtype}: {desc} ({direction})")
    return "\n".join(lines)


def get_entity_types_for_prompt() -> List[dict]:
    cfg = _load_config()
    return [{"type": t, "description": info["description"]} for t, info in cfg["entity_types"].items()]


# ------------------------------------------------------------------
# Edge type helpers
# ------------------------------------------------------------------

def get_structural_edge_types() -> Set[str]:
    return set(_load_config()["relationship_types"]["structural"]["types"])


def get_semantic_edge_types() -> Set[str]:
    semantic = _load_config()["relationship_types"]["semantic"]["types"]
    if isinstance(semantic, dict):
        return set(semantic.keys())
    return set(semantic)


def get_prioritized_edge_types() -> Set[str]:
    return set(_load_config()["relationship_types"]["prioritized"]["types"])


def normalize_relationship_type(raw_type: str) -> str:
    """Normalize a relationship type using the config map.

    "INVOLVED_IN" -> "WORKS_ON", unknown -> pass-through.
    """
    upper = raw_type.upper().replace(" ", "_")
    cfg = _load_config()
    norm_map = cfg.get("relationship_normalization", {})
    if upper in norm_map:
        return norm_map[upper]
    flip_map = cfg.get("relationship_direction_flips", {})
    if upper in flip_map:
        return flip_map[upper]["standard"]
    return upper


def should_flip_relationship(raw_type: str) -> bool:
    """Check if a relationship type requires source/target to be swapped."""
    upper = raw_type.upper().replace(" ", "_")
    flip_map = _load_config().get("relationship_direction_flips", {})
    return upper in flip_map


# ------------------------------------------------------------------
# Node type helpers
# ------------------------------------------------------------------

def get_entity_node_types() -> Set[str]:
    return get_all_entity_types()


def get_internal_node_types() -> Set[str]:
    return set(_load_config().get("internal_node_types", []))


# ------------------------------------------------------------------
# Entity catalog (alias resolution)
# ------------------------------------------------------------------

def load_catalog(catalog_path: str) -> dict:
    """Load entity catalog from disk."""
    global _catalog, _catalog_path
    _catalog_path = catalog_path
    p = Path(catalog_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            _catalog = json.load(f)
        logger.info(f"Loaded entity catalog: {len(_catalog.get('entities', {}))} entities")
    else:
        _catalog = {"entities": {}, "metadata": {}}
    return _catalog


def get_catalog() -> dict:
    if _catalog is None:
        return {"entities": {}, "metadata": {}}
    return _catalog


def _get_static_name_aliases() -> Dict[str, str]:
    """Load entity_name_aliases from config (case-insensitive lookup map)."""
    cfg = _load_config()
    aliases = cfg.get("entity_name_aliases", {})
    # Build case-insensitive map
    return {k.lower(): v for k, v in aliases.items()}


def resolve_entity_name(name: str) -> str:
    """Resolve a known alias to its standard entity name.

    Checks in order:
    1. Static aliases from entity_config.json (e.g., "Acme Corporation" → "Acme")
    2. Catalog aliases (from previous Gold runs)
    Returns standard name if found, otherwise pass-through.
    """
    # 1. Static aliases from config
    static = _get_static_name_aliases()
    canonical = static.get(name.lower())
    if canonical:
        return canonical

    # 2. Catalog aliases
    cat = get_catalog()
    entities = cat.get("entities", {})

    # Direct match
    if name in entities:
        return entities[name].get("standard_name", name)

    # Alias lookup (case-insensitive)
    name_lower = name.lower()
    for std_name, info in entities.items():
        if std_name.lower() == name_lower:
            return info.get("standard_name", std_name)
        for alias in info.get("aliases", []):
            if alias.lower() == name_lower:
                return info.get("standard_name", std_name)

    return name


def expand_entity_aliases(name: str) -> List[str]:
    """Expand an entity name to include all known aliases (bidirectional).

    "Berlin Office" → ["Berlin Office", "BER", "Ber"]
    "BER" → ["BER", "Berlin Office", "Ber"]
    Unknown name → [name] (just itself)
    """
    cat = get_catalog()
    entities = cat.get("entities", {})
    name_lower = name.lower()

    # Check if name is a standard name
    for std_name, info in entities.items():
        if std_name.lower() == name_lower:
            result = [std_name]
            for alias in info.get("aliases", []):
                if alias not in result:
                    result.append(alias)
            return result

    # Check if name is an alias
    for std_name, info in entities.items():
        for alias in info.get("aliases", []):
            if alias.lower() == name_lower:
                result = [name, std_name]
                for a in info.get("aliases", []):
                    if a not in result:
                        result.append(a)
                return result

    return [name]


def save_catalog(catalog: dict, catalog_path: Optional[str] = None):
    """Save entity catalog to disk."""
    path = catalog_path or _catalog_path
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved entity catalog: {len(catalog.get('entities', {}))} entities")
