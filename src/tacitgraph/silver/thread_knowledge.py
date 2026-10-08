"""Knowledge-graph extraction for Silver chunks: entities, relationships and participant handling."""

import logging
from typing import Any

from tacitgraph.silver.kg_entity_extractor import (
    KGEntity,
)
from tacitgraph.silver.thread_storage import ThreadStorage

logger = logging.getLogger(__name__)


class ThreadKnowledgeExtraction(ThreadStorage):
    """Knowledge-graph extraction for Silver chunks: entities, relationships and participant handling."""

    def _extract_kg_entities(
        self, text: str, language: str, chunk_id: str = ""
    ) -> tuple[list[dict[str, Any]], list[KGEntity], str, str]:
        """
        Extract knowledge graph entities using modular extractor.

        Delegates to configured KG extractor (spaCy, LLM, or hybrid).
        In LLM mode, also returns text_english and detected source_language
        from the same API call (combined KG extraction + translation).

        Returns tuple of (entity_dicts, raw_entities, text_english, detected_language).
        For spaCy/local mode, text_english and detected_language are empty strings
        (caller falls back to using anonymized_text as-is).
        """
        try:
            entities = self.kg_extractor.extract(text, language)
        except Exception as e:
            error_str = str(e)
            is_content_filter = "content_filter" in error_str
            if is_content_filter:
                self.stats["content_filter_blocks"] += 1
            self.stats["llm_failures"].append(
                {
                    "chunk_id": chunk_id,
                    "stage": "entity_extraction",
                    "reason": "content_filter" if is_content_filter else "error",
                    "message": error_str[:200],
                }
            )
            logger.error(f"KG entity extraction failed for chunk_id={chunk_id}: {e}")
            return [], [], "", ""

        # Deduplicate entities by (name, type) — keep first occurrence
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.entity.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        entity_dicts = [entity.to_dict() for entity in unique_entities]
        self.stats["kg_entities_extracted"] += len(entity_dicts)

        # LLM extractor returns text_english and source_language from the same call
        text_english = getattr(self.kg_extractor, "last_text_english", "") or ""
        detected_lang = getattr(self.kg_extractor, "last_source_language", "") or ""

        return entity_dicts, unique_entities, text_english, detected_lang

    def _extract_kg_relationships(
        self,
        text: str,
        entities: list[KGEntity],
        language: str,
        chunk_id: str = "",
    ) -> list[dict[str, Any]]:
        """
        Extract relationships between entities using modular extractor.

        Delegates to configured relationship extractor (cooccurrence, LLM, hybrid).
        Returns list of relationship dicts for PathRAG knowledge graph.
        """
        if not self.extract_relationships or not self.relationship_extractor:
            return []

        if len(entities) < 2:
            return []

        try:
            relationships = self.relationship_extractor.extract(text, entities, language)
        except Exception as e:
            error_str = str(e)
            is_content_filter = "content_filter" in error_str
            if is_content_filter:
                self.stats["content_filter_blocks"] += 1
            self.stats["llm_failures"].append(
                {
                    "chunk_id": chunk_id,
                    "stage": "relationship_extraction",
                    "reason": "content_filter" if is_content_filter else "error",
                    "message": error_str[:200],
                }
            )
            logger.error(f"KG relationship extraction failed for chunk_id={chunk_id}: {e}")
            return []

        relationship_dicts = [r.to_dict() for r in relationships]
        self.stats["kg_relationships_extracted"] += len(relationship_dicts)
        return relationship_dicts

    def _get_person_pseudonym(self, name: str) -> str:
        """
        Get a pseudonym for a person name.

        Lookup order:
        1. Identity registry — known email senders/recipients (exact + fuzzy)
        2. Anonymizer's value mapping — consistent within processing run
        3. Anonymizer pass — generates new PERSON_N and caches it

        Known people (registry) get stable pseudonyms tied to their email.
        NER-only names get consistent pseudonyms via the anonymizer's
        _value_mapping (same text → same pseudonym within a run).
        """
        # 1. Identity registry (email-based, stable across runs)
        if self.identity_registry:
            pseudonym = self.identity_registry.get_pseudonym(name)
            if pseudonym:
                return pseudonym

        # 2. Anonymizer's existing mapping (consistent within run)
        mapped = self.anonymizer._value_mapping.get(name, "")
        if mapped:
            return mapped.strip("[]")

        # 3. Run anonymizer — detects as PERSON PII, generates & caches pseudonym
        anon_result = self.anonymizer.anonymize(name, "en")
        if anon_result.anonymized_text != name:
            return anon_result.anonymized_text.strip("[]")

        # 4. Anonymizer didn't detect it as PII — force a consistent pseudonym
        if not hasattr(self, "_ner_person_counter"):
            self._ner_person_counter = 900  # high offset to avoid collision
        self._ner_person_counter += 1
        pseudonym = f"PERSON_{self._ner_person_counter:03d}"
        self.anonymizer._value_mapping[name] = f"[{pseudonym}]"
        return pseudonym

    def _anonymize_kg_entities(self, entity_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Replace PERSON entity text with pseudonyms in KG entity dicts.

        All PERSON entities are anonymized regardless of is_pii flag,
        since person names in the knowledge graph are always PII.
        """
        anonymized = []
        for e in entity_dicts:
            e_copy = dict(e)
            if e_copy.get("type") == "PERSON":
                name = e_copy.get("entity", e_copy.get("text", ""))
                pseudonym = self._get_person_pseudonym(name)
                e_copy["entity"] = pseudonym
                e_copy["original_entity"] = name
            anonymized.append(e_copy)
        return anonymized

    def _anonymize_kg_relationships(self, rel_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Replace PERSON source/target names with pseudonyms in KG relationship dicts.
        """
        anonymized = []
        for r in rel_dicts:
            r_copy = dict(r)
            for role in ("source", "target"):
                name = r_copy.get(role, "")
                role_type = r_copy.get(f"{role}_type", "")
                if role_type in ("PERSON", "person") and name:
                    r_copy[role] = self._get_person_pseudonym(name)
            anonymized.append(r_copy)
        return anonymized

    def _anonymize_participants(self, participants: list[str]) -> list[str]:
        """
        Anonymize a list of participant names using identity registry.
        """
        result = []
        for p in participants:
            pseudonym = None
            if self.identity_registry:
                pseudonym = self.identity_registry.get_pseudonym(p)
            if pseudonym:
                result.append(pseudonym)
            else:
                # Fall back to full anonymizer (handles emails, etc.)
                anon = self.anonymizer.anonymize(p, "en")
                result.append(anon.anonymized_text)
        return result

    def _anonymize_participant(self, participant: str) -> str:
        """Anonymize participant name"""
        # Use the same anonymizer for consistency
        result = self.anonymizer.anonymize(participant, "en")
        return result.anonymized_text
