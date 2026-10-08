"""BM25 keyword index over Silver chunks.

Sentence embeddings place "who is <name>?" near any email that introduces a
person, whether or not it mentions <name>. Rare tokens such as names, ticket
numbers and product codes are exactly what BM25 ranks well, so the vector and
keyword rankings are fused with Reciprocal Rank Fusion.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = (
    "a about all also an and any are as at be been but by can could did do does "
    "for from had has have he her his how i if in into is it its me my no not of "
    "on or our she so than that the their them then there these they this to was "
    "we were what when where which who whom why will with would you your "
    "tell know give show find list explain describe anything something someone"
)
STOPWORDS = frozenset(_STOPWORDS.split())


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stopwords or single characters."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over an in-memory corpus of ``(doc_id, text)`` pairs."""

    def __init__(self, docs: Iterable[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.term_freqs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        doc_freq: Counter[str] = Counter()
        for doc_id, text in docs:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.doc_ids.append(doc_id)
            self.term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            doc_freq.update(tf.keys())
        n = len(self.doc_ids)
        self.avg_len = (sum(self.doc_lens) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def __len__(self) -> int:
        return len(self.doc_ids)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, score)`` pairs with a positive score."""
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        if not terms or not self.doc_ids:
            return []
        scores: list[tuple[str, float]] = []
        for doc_id, tf, length in zip(self.doc_ids, self.term_freqs, self.doc_lens, strict=True):
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                    score += self.idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(*rankings: list[str], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists; ``score = sum(1 / (k + rank))`` over the lists containing the id."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
