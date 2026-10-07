"""
Semantic Chunker

Splits documents into semantically meaningful chunks for embedding
and retrieval. Supports multiple chunking strategies.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from enum import Enum

import tiktoken


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"           # Fixed token count
    SENTENCE = "sentence"               # Sentence-based
    PARAGRAPH = "paragraph"             # Paragraph-based
    SEMANTIC = "semantic"               # Semantic similarity-based
    RECURSIVE = "recursive"             # Recursive character splitting


@dataclass
class Chunk:
    """A text chunk with metadata"""

    # Identification
    chunk_id: str
    doc_id: str
    chunk_index: int                    # Position in document

    # Content
    text: str
    token_count: int

    # Position
    start_char: int
    end_char: int

    # Context
    overlap_before: str = ""            # Overlapping text from previous chunk
    overlap_after: str = ""             # Overlapping text for next chunk

    # Metadata (inherited from document)
    source_file: Optional[str] = None
    source_date: Optional[datetime] = None
    page_number: Optional[int] = None

    # Processing metadata
    language: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "source_file": self.source_file,
            "source_date": self.source_date.isoformat() if self.source_date else None,
            "page_number": self.page_number,
            "language": self.language,
        }

    def get_text_with_overlap(self) -> str:
        """Get text including overlap context"""
        parts = []
        if self.overlap_before:
            parts.append(f"[...] {self.overlap_before}")
        parts.append(self.text)
        if self.overlap_after:
            parts.append(f"{self.overlap_after} [...]")
        return " ".join(parts)


class SemanticChunker:
    """
    Split documents into chunks for embedding and retrieval.

    Supports multiple strategies and handles overlap to maintain
    context across chunk boundaries.
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Paragraph patterns
    PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        encoding_name: str = "cl100k_base"  # GPT-4/text-embedding-3 encoding
    ):
        """
        Initialize the chunker.

        Args:
            strategy: Chunking strategy to use
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size (discard smaller)
            encoding_name: Tiktoken encoding name
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback: approximate with word count
            self.encoding = None

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            doc_id: Document ID
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # Choose chunking method
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentence(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraph(text)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(text)
        else:  # RECURSIVE
            chunks = self._chunk_recursive(text)

        # Convert to Chunk objects with metadata
        result = []
        for i, (chunk_text, start, end) in enumerate(chunks):
            token_count = self._count_tokens(chunk_text)

            # Generate chunk ID
            chunk_id = self._generate_chunk_id(doc_id, i, chunk_text)

            # Add overlap context
            overlap_before = self._get_overlap_before(text, start)
            overlap_after = self._get_overlap_after(text, end)

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=i,
                text=chunk_text,
                token_count=token_count,
                start_char=start,
                end_char=end,
                overlap_before=overlap_before,
                overlap_after=overlap_after,
                source_file=metadata.get("source_file"),
                source_date=metadata.get("source_date"),
                page_number=metadata.get("page_number"),
                language=metadata.get("language"),
            )
            result.append(chunk)

        return result

    def chunk_with_pages(
        self,
        pages: List[str],
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk document with page information preserved.

        Args:
            pages: List of page texts
            doc_id: Document ID
            metadata: Optional metadata

        Returns:
            List of chunks with page numbers
        """
        all_chunks = []
        chunk_index = 0

        for page_num, page_text in enumerate(pages, 1):
            if not page_text.strip():
                continue

            page_metadata = {
                **(metadata or {}),
                "page_number": page_num
            }

            page_chunks = self.chunk(page_text, doc_id, page_metadata)

            # Update chunk indices to be global
            for chunk in page_chunks:
                chunk.chunk_index = chunk_index
                chunk.chunk_id = self._generate_chunk_id(doc_id, chunk_index, chunk.text)
                chunk_index += 1

            all_chunks.extend(page_chunks)

        return all_chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Approximate: ~4 chars per token
            return len(text) // 4

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate unique chunk ID"""
        unique_str = f"{doc_id}:{index}:{text[:100]}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    # =========================================================================
    # Chunking Strategies
    # =========================================================================

    def _chunk_fixed_size(self, text: str) -> List[tuple]:
        """
        Fixed-size chunking based on token count.

        Returns list of (text, start_char, end_char)
        """
        chunks = []

        if self.encoding:
            tokens = self.encoding.encode(text)
            total_tokens = len(tokens)

            start_token = 0
            while start_token < total_tokens:
                end_token = min(start_token + self.chunk_size, total_tokens)

                # Decode chunk
                chunk_tokens = tokens[start_token:end_token]
                chunk_text = self.encoding.decode(chunk_tokens)

                # Calculate character positions (approximate)
                start_char = len(self.encoding.decode(tokens[:start_token]))
                end_char = start_char + len(chunk_text)

                chunks.append((chunk_text, start_char, end_char))

                # Move forward with overlap
                start_token = end_token - self.chunk_overlap

        else:
            # Fallback: character-based
            char_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4

            start = 0
            while start < len(text):
                end = min(start + char_size, len(text))
                chunks.append((text[start:end], start, end))
                start = end - char_overlap

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[tuple]:
        """
        Sentence-based chunking.

        Keeps sentences intact while respecting chunk size limits.
        """
        # Split into sentences
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        current_start = 0
        char_pos = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If sentence alone exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, current_start, char_pos))
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence
                sub_chunks = self._chunk_fixed_size(sentence)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, char_pos + sub_start, char_pos + sub_end))

                current_start = char_pos + len(sentence) + 1

            elif current_tokens + sentence_tokens > self.chunk_size:
                # Chunk is full, start new one
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, current_start, char_pos))

                # Start new chunk (with overlap)
                overlap_sentences = []
                overlap_tokens = 0

                for s in reversed(current_chunk):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
                current_start = char_pos - sum(len(s) + 1 for s in overlap_sentences)

            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

            char_pos += len(sentence) + 1  # +1 for space/newline

        # Flush remaining
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, current_start, len(text)))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[tuple]:
        """
        Paragraph-based chunking.

        Keeps paragraphs intact while respecting chunk size limits.
        """
        # Split into paragraphs
        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        current_start = 0
        char_pos = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # If paragraph alone exceeds chunk size, split by sentences
            if para_tokens > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append((chunk_text, current_start, char_pos))
                    current_chunk = []
                    current_tokens = 0
                    current_start = char_pos

                # Split paragraph by sentences
                para_chunks = self._chunk_by_sentence(para)
                for sub_text, sub_start, sub_end in para_chunks:
                    chunks.append((sub_text, char_pos + sub_start, char_pos + sub_end))

                current_start = char_pos + len(para) + 2

            elif current_tokens + para_tokens > self.chunk_size:
                # Chunk is full
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append((chunk_text, current_start, char_pos))

                current_chunk = [para]
                current_tokens = para_tokens
                current_start = char_pos

            else:
                current_chunk.append(para)
                current_tokens += para_tokens

            char_pos += len(para) + 2  # +2 for paragraph break

        # Flush remaining
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append((chunk_text, current_start, len(text)))

        return chunks

    def _chunk_recursive(self, text: str) -> List[tuple]:
        """
        Recursive character splitting.

        Tries to split on semantic boundaries in order:
        1. Double newlines (paragraphs)
        2. Single newlines
        3. Sentences
        4. Spaces (words)
        5. Characters
        """
        separators = ["\n\n", "\n", ". ", " ", ""]

        return self._recursive_split(text, separators, 0)

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        start_offset: int
    ) -> List[tuple]:
        """Recursively split text"""
        # Base case: text fits in chunk
        if self._count_tokens(text) <= self.chunk_size:
            if text.strip():
                return [(text, start_offset, start_offset + len(text))]
            return []

        # Try each separator
        for sep in separators:
            if sep == "":
                # Last resort: character split
                return self._chunk_fixed_size(text)

            if sep not in text:
                continue

            # Split on separator
            parts = text.split(sep)
            chunks = []
            current_chunk = []
            current_tokens = 0
            current_start = start_offset
            char_pos = 0

            for i, part in enumerate(parts):
                part_tokens = self._count_tokens(part)

                # Part alone is too big - recurse with next separator
                if part_tokens > self.chunk_size:
                    # Flush current
                    if current_chunk:
                        chunk_text = sep.join(current_chunk)
                        chunks.append((chunk_text, current_start, start_offset + char_pos))
                        current_chunk = []
                        current_tokens = 0

                    # Recurse on large part
                    sub_chunks = self._recursive_split(
                        part,
                        separators[separators.index(sep) + 1:],
                        start_offset + char_pos
                    )
                    chunks.extend(sub_chunks)
                    current_start = start_offset + char_pos + len(part) + len(sep)

                elif current_tokens + part_tokens > self.chunk_size:
                    # Chunk full
                    if current_chunk:
                        chunk_text = sep.join(current_chunk)
                        chunks.append((chunk_text, current_start, start_offset + char_pos))

                    current_chunk = [part]
                    current_tokens = part_tokens
                    current_start = start_offset + char_pos

                else:
                    current_chunk.append(part)
                    current_tokens += part_tokens

                char_pos += len(part) + len(sep)

            # Flush remaining
            if current_chunk:
                chunk_text = sep.join(current_chunk)
                chunks.append((chunk_text, current_start, start_offset + len(text)))

            if chunks:
                return chunks

        # Fallback
        return self._chunk_fixed_size(text)

    def _chunk_semantic(self, text: str) -> List[tuple]:
        """
        Semantic chunking using embedding similarity.

        Requires sentence-transformers for embeddings.
        Falls back to recursive if not available.
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Split into sentences
            sentences = self.SENTENCE_ENDINGS.split(text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                return self._chunk_recursive(text)

            # Get embeddings
            embeddings = model.encode(sentences)

            # Find breakpoints based on similarity
            breakpoints = [0]
            for i in range(1, len(sentences)):
                similarity = np.dot(embeddings[i-1], embeddings[i])
                # Low similarity = semantic break
                if similarity < 0.5:
                    breakpoints.append(i)

            breakpoints.append(len(sentences))

            # Create chunks from breakpoints
            chunks = []
            char_pos = 0

            for i in range(len(breakpoints) - 1):
                start_idx = breakpoints[i]
                end_idx = breakpoints[i + 1]

                chunk_sentences = sentences[start_idx:end_idx]
                chunk_text = " ".join(chunk_sentences)

                # Split if too large
                if self._count_tokens(chunk_text) > self.chunk_size:
                    sub_chunks = self._chunk_by_sentence(chunk_text)
                    for sub_text, sub_start, sub_end in sub_chunks:
                        chunks.append((sub_text, char_pos + sub_start, char_pos + sub_end))
                else:
                    start_char = sum(len(s) + 1 for s in sentences[:start_idx])
                    end_char = sum(len(s) + 1 for s in sentences[:end_idx])
                    chunks.append((chunk_text, start_char, end_char))

                char_pos = sum(len(s) + 1 for s in sentences[:end_idx])

            return chunks

        except ImportError:
            # Fallback to recursive
            return self._chunk_recursive(text)

    def _get_overlap_before(self, text: str, start: int) -> str:
        """Get overlap text from before the chunk"""
        if start <= 0:
            return ""

        # Get text before start
        overlap_chars = self.chunk_overlap * 4  # Approximate
        overlap_start = max(0, start - overlap_chars)
        overlap_text = text[overlap_start:start].strip()

        # Truncate to last sentence boundary if possible
        sentences = self.SENTENCE_ENDINGS.split(overlap_text)
        if len(sentences) > 1:
            overlap_text = sentences[-1]

        return overlap_text

    def _get_overlap_after(self, text: str, end: int) -> str:
        """Get overlap text from after the chunk"""
        if end >= len(text):
            return ""

        # Get text after end
        overlap_chars = self.chunk_overlap * 4
        overlap_end = min(len(text), end + overlap_chars)
        overlap_text = text[end:overlap_end].strip()

        # Truncate to first sentence boundary if possible
        sentences = self.SENTENCE_ENDINGS.split(overlap_text)
        if len(sentences) > 1:
            overlap_text = sentences[0]

        return overlap_text


# Convenience function
def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 1024,
    overlap: int = 50
) -> List[Chunk]:
    """
    Chunk text with default settings.

    Args:
        text: Text to chunk
        doc_id: Document ID
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks

    Returns:
        List of Chunk objects
    """
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return chunker.chunk(text, doc_id)
