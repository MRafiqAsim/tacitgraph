"""Query understanding: keyword and entity extraction and matching against the graph."""

import json
import logging
import os
from pathlib import Path

from tacitgraph.prompt_loader import format_prompt, get_prompt

logger = logging.getLogger(__name__)


class QueryAnalysis:
    """Query understanding: keyword and entity extraction and matching against the graph."""

    def _extract_query_entities(self, query: str, top_n: int = 10) -> tuple[list[str], list[str]]:
        """
        PathRAG Node Retrieval (Stage 1 per paper).

        Per-keyword matching: each original keyword gets its own best entity match,
        guaranteeing diverse representation. Then remaining slots are filled from
        expanded aliases.

        Returns:
            Tuple of (entity_ids, entity_names)
        """
        # Extract original keywords (before expansion)
        if self.mode == "llm":
            original_keywords = self._extract_keywords_llm(query)
        else:
            original_keywords = self._extract_keywords_local(query)

        if not original_keywords:
            return [], []

        logger.info(f"Original keywords: {original_keywords}")

        # Per-keyword matching: find best entity for EACH keyword individually
        # This guarantees "Lisbon" gets a slot even when "migration" has higher scores
        matched = []
        seen_ids = set()
        seen_names = set()

        for kw in original_keywords:
            # Match this single keyword
            kw_matches = self.toolkit.node_retrieval([kw], top_n=3)
            for entity_id, name in kw_matches:
                if entity_id not in seen_ids and name.lower() not in seen_names:
                    matched.append((entity_id, name))
                    seen_ids.add(entity_id)
                    seen_names.add(name.lower())
                    logger.info(f"  '{kw}' → {name}")
                    break  # one entity per keyword

        logger.info(
            f"Per-keyword matches: {len(matched)} entities from {len(original_keywords)} keywords"
        )

        # Fill remaining slots from expanded aliases
        if len(matched) < top_n:
            from tacitgraph.entity_registry import expand_entity_aliases

            expanded = []
            for kw in original_keywords:
                expanded.extend(expand_entity_aliases(kw))
            all_keywords = list(set(expanded))

            expanded_matches = self.toolkit.node_retrieval(all_keywords, top_n=top_n * 2)
            for entity_id, name in expanded_matches:
                if entity_id not in seen_ids and name.lower() not in seen_names:
                    matched.append((entity_id, name))
                    seen_ids.add(entity_id)
                    seen_names.add(name.lower())
                    if len(matched) >= top_n:
                        break

        matched = matched[:top_n]
        if matched:
            entity_ids = [m[0] for m in matched]
            entity_names = [m[1] for m in matched]
            logger.info(f"PathRAG node retrieval: {len(entity_ids)} entities: {entity_names}")
            return entity_ids, entity_names

        return [], []

    def _extract_keywords(self, query: str) -> list[str]:
        """
        Extract keywords from query with alias expansion.

        LLM mode: GPT-4o extracts entity names and keywords.
        Local mode: spaCy NER + content-word heuristics.
        Both modes: resolve aliases via entity catalog.

        Note: _extract_query_entities uses _extract_keywords_llm/_extract_keywords_local
        directly for two-phase matching. This method is used by other callers that
        need the full expanded keyword list.
        """
        if self.mode == "llm":
            keywords = self._extract_keywords_llm(query)
        else:
            keywords = self._extract_keywords_local(query)

        from tacitgraph.entity_registry import expand_entity_aliases

        logger.info(f"Keywords before expansion: {keywords}")
        expanded = []
        for kw in keywords:
            expanded.extend(expand_entity_aliases(kw))
        result = list(set(expanded))
        logger.info(f"Keywords after expansion: {result}")
        return result

    def _extract_keywords_llm(self, query: str) -> list[str]:
        """Extract keywords using GPT-4o."""
        import json as _json

        try:
            from tacitgraph.entity_registry import get_all_entity_types

            entity_types = list(get_all_entity_types())
            prompt_template = get_prompt("retrieval", "query_keyword_extraction", "user_prompt")
            system_prompt = format_prompt(
                get_prompt("retrieval", "query_keyword_extraction", "system_prompt"),
                entity_types=entity_types,
            )
            user_prompt = format_prompt(prompt_template, query=query)

            from tacitgraph.llm_response import extract_llm_content

            if not self.llm_client:
                return self._extract_keywords_local(query)
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

            response = self.llm_client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=200,
            )

            content = extract_llm_content(response, context="keyword extraction")

            import re

            if content.startswith("```"):
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    content = match.group()

            keywords = _json.loads(content)
            if isinstance(keywords, list):
                return [str(k) for k in keywords if k]
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed, falling back to local: {e}")

        return self._extract_keywords_local(query)

    def _extract_keywords_local(self, query: str) -> list[str]:
        """Extract keywords using spaCy NER + content word heuristics."""
        import re

        keywords = []

        # 1. Quoted phrases (highest priority — explicit user intent)
        quoted = re.findall(r'"([^"]+)"', query)
        keywords.extend(quoted)

        # 2. spaCy NER extraction (local, no LLM needed)
        try:
            import spacy

            if not hasattr(self, "_nlp"):
                self._nlp = spacy.load("en_core_web_sm")
            doc = self._nlp(query)
            for ent in doc.ents:
                keywords.append(ent.text)
        except Exception:
            pass  # spaCy not available, fall through to heuristics

        # 3. Content words (capitalized, acronyms, meaningful terms)
        stop_words = {
            "the",
            "a",
            "an",
            "what",
            "who",
            "where",
            "when",
            "how",
            "why",
            "which",
            "did",
            "does",
            "do",
            "is",
            "are",
            "was",
            "were",
            "and",
            "or",
            "but",
            "about",
            "tell",
            "me",
            "can",
            "you",
            "find",
            "show",
            "get",
            "give",
            "provide",
            "list",
            "all",
            "every",
            "any",
            "some",
            "project",
            "projects",
            "email",
            "emails",
            "thread",
            "threads",
            "information",
            "details",
            "data",
            "discussed",
            "mentioned",
            "people",
            "person",
            "things",
            "stuff",
            "work",
            "used",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "from",
            "by",
        }

        words = query.split()
        content_words = []
        for word in words:
            clean = re.sub(r"[?.!,;:]$", "", word)
            if clean.lower() in stop_words or len(clean) < 2:
                continue
            content_words.append(clean)
            # Capitalized words and acronyms as separate keywords
            if (clean.isupper() and len(clean) >= 2) or (clean[0:1].isupper() and len(clean) > 1):
                keywords.append(clean)

        # 4. Bigrams/trigrams from content words
        if len(content_words) >= 2:
            for i in range(len(content_words)):
                for j in range(i + 2, min(i + 4, len(content_words) + 1)):
                    keywords.append(" ".join(content_words[i:j]))

        # Add individual content words
        keywords.extend(content_words)

        return list(set(keywords))

    def _match_graph_entities_by_name(self, candidates: list[str]) -> list[str]:
        """Fallback: match candidate strings against known graph entity names."""
        if not self.gold_path:
            return candidates

        # Load graph entity names (cached)
        if not hasattr(self, "_graph_entity_names"):
            self._graph_entity_names = {}
            graph_file = Path(self.gold_path) / "knowledge_graph" / "nodes.json"
            if graph_file.exists():
                try:
                    with open(graph_file, encoding="utf-8") as f:
                        graph_data = json.load(f)
                    for node_id, node in graph_data.items():
                        name = node.get("name", node_id)
                        node_type = node.get("node_type", node.get("type", ""))
                        if name and node_type not in ("CHUNK", "THREAD", ""):
                            self._graph_entity_names[name.lower()] = name
                except Exception:
                    pass

        if not self._graph_entity_names:
            return candidates

        matched = []
        for candidate in candidates:
            candidate_lower = candidate.lower()
            if candidate_lower in self._graph_entity_names:
                matched.append(self._graph_entity_names[candidate_lower])
                continue
            for graph_lower, graph_name in self._graph_entity_names.items():
                if candidate_lower in graph_lower or graph_lower in candidate_lower:
                    matched.append(graph_name)

        return list(set(matched))

    def _detect_aggregate_type(self, query: str) -> list[str] | None:
        """Detect if query asks for a list of entities and return matching entity types.

        Returns a list of entity types to try (primary + related fallbacks).
        For example, 'projects' → ['PRODUCT', 'DOCUMENT', 'CONCEPT'] because
        emails may label work items differently than the user's terminology.
        """
        import re

        query_lower = query.lower()

        # Map user terms to primary + fallback entity types
        type_patterns = {
            r"(project|initiative|task|work item|topic|subject)": [
                "PRODUCT",
                "DOCUMENT",
                "CONCEPT",
            ],
            r"(product|system|tool|software|application|platform)": ["PRODUCT"],
            r"(people|person|employee|team member|participant|who)": ["PERSON"],
            r"(organization|company|department|team|vendor|supplier|client)": ["ORG"],
            r"(event|meeting|milestone|deadline|conference)": ["EVENT"],
            r"(document|report|file|attachment|specification)": ["DOCUMENT"],
            r"(location|city|country|office|place|region)": ["GPE"],
        }

        # Check for listing/aggregate intent
        if not re.search(
            r"(list|all|every|what\b.*\b(are|names|types)|provide|show|give me|how many|which|discussed|mentioned)",
            query_lower,
        ):
            return None

        for pattern, entity_types in type_patterns.items():
            if re.search(pattern, query_lower):
                return entity_types

        return None

    def _get_all_thread_subjects(self) -> list[str]:
        """Get all unique thread subjects from silver layer.

        NOTE: Currently unused. Injecting all subjects as LLM context caused
        the model to surface wrong emails based on subject-line keyword matches
        rather than relying on scored retrieval results.
        """
        if not self.silver_path:
            return []

        subjects = set()
        silver = Path(self.silver_path)
        for folder in [
            "not_personal/email_chunks",
            "not_personal/attachment_chunks",
            "not_personal/document_chunks",
        ]:
            folder_path = silver / folder
            if not folder_path.exists():
                continue
            for f in folder_path.glob("*.json"):
                try:
                    with open(f, encoding="utf-8") as fh:
                        data = json.load(fh)
                    subj = data.get("thread_subject", "")
                    if subj:
                        subjects.add(subj)
                except Exception:
                    pass
        return list(subjects)
