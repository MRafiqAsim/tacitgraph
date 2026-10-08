"""Token-aware chunking strategies (no embedding models involved)."""

import pytest

from tacitgraph.silver.chunker import Chunk, ChunkingStrategy, SemanticChunker, chunk_text

SENTENCE = "Jane Doe shared the Project Atlas release plan with the Lisbon office."


def paragraphs(count: int, sentences_per_paragraph: int = 3) -> str:
    return "\n\n".join(
        " ".join(
            f"Paragraph {p} sentence {s} covers {SENTENCE}" for s in range(sentences_per_paragraph)
        )
        for p in range(count)
    )


@pytest.fixture
def small_chunker():
    return SemanticChunker(chunk_size=60, chunk_overlap=10)


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_empty_text_yields_no_chunks(small_chunker, text):
    assert small_chunker.chunk(text, "doc") == []


def test_short_text_is_a_single_chunk():
    chunks = SemanticChunker().chunk(SENTENCE, "doc-1")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == SENTENCE
    assert (chunk.chunk_index, chunk.start_char, chunk.end_char) == (0, 0, len(SENTENCE))
    assert chunk.overlap_before == chunk.overlap_after == ""


@pytest.mark.parametrize(
    "strategy",
    [ChunkingStrategy.RECURSIVE, ChunkingStrategy.PARAGRAPH, ChunkingStrategy.SENTENCE],
)
def test_long_text_respects_chunk_size(strategy):
    chunker = SemanticChunker(strategy=strategy, chunk_size=60, chunk_overlap=10)
    chunks = chunker.chunk(paragraphs(6), "doc")
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(0 < c.token_count <= 60 for c in chunks)


def test_recursive_strategy_prefers_paragraph_boundaries(small_chunker):
    text = paragraphs(4, sentences_per_paragraph=1)
    chunks = small_chunker.chunk(text, "doc")
    for chunk in chunks:
        # end_char of a flushed chunk also spans the trailing separator
        assert text[chunk.start_char : chunk.end_char].strip() == chunk.text
        assert not chunk.text.startswith("\n")


def test_fixed_size_without_overlap_covers_all_tokens():
    chunker = SemanticChunker(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=10, chunk_overlap=0)
    text = "word " * 40
    chunks = chunker.chunk(text, "doc")
    assert sum(c.token_count for c in chunks) == chunker._count_tokens(text)
    assert "".join(c.text for c in chunks) == text


def test_sentence_strategy_keeps_sentences_whole():
    chunker = SemanticChunker(strategy=ChunkingStrategy.SENTENCE, chunk_size=40, chunk_overlap=0)
    sentences = [f"Sentence {i} mentions Project Atlas and the Berlin office." for i in range(8)]
    chunks = chunker.chunk(" ".join(sentences), "doc")
    for chunk in chunks:
        assert chunk.text.endswith(".")
        assert chunk.text.split(". ")[0] + "." in sentences or chunk.text in sentences


def test_overlap_context_is_attached_to_neighbouring_chunks(small_chunker):
    chunks = small_chunker.chunk(paragraphs(6), "doc")
    assert chunks[0].overlap_before == ""
    assert chunks[0].overlap_after
    assert chunks[1].overlap_before
    assert chunks[-1].overlap_after == ""


def test_chunk_ids_are_deterministic_and_unique(small_chunker):
    text = paragraphs(5)
    first = [c.chunk_id for c in small_chunker.chunk(text, "doc")]
    second = [c.chunk_id for c in small_chunker.chunk(text, "doc")]
    other_doc = [c.chunk_id for c in small_chunker.chunk(text, "another-doc")]
    assert first == second
    assert len(set(first)) == len(first)
    assert set(first).isdisjoint(other_doc)


def test_metadata_is_propagated():
    chunk = SemanticChunker().chunk(
        SENTENCE, "doc", {"source_file": "atlas.pdf", "language": "en", "page_number": 2}
    )[0]
    assert (chunk.source_file, chunk.language, chunk.page_number) == ("atlas.pdf", "en", 2)


def test_chunk_with_pages_numbers_chunks_globally():
    chunker = SemanticChunker(chunk_size=60, chunk_overlap=10)
    pages = [paragraphs(3), "", paragraphs(3)]
    chunks = chunker.chunk_with_pages(pages, "doc")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 3}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_serialization_and_overlap_text():
    chunk = Chunk(
        chunk_id="c1",
        doc_id="doc",
        chunk_index=0,
        text="middle",
        token_count=1,
        start_char=10,
        end_char=16,
        overlap_before="before",
        overlap_after="after",
    )
    assert chunk.get_text_with_overlap() == "[...] before middle after [...]"
    data = chunk.to_dict()
    assert data["chunk_id"] == "c1"
    assert data["source_date"] is None


def test_chunk_text_convenience_function():
    chunks = chunk_text(paragraphs(6), "doc", chunk_size=60, overlap=10)
    assert len(chunks) > 1
    assert all(c.doc_id == "doc" for c in chunks)


# --- Regression: overlapping fixed-size windows used to loop forever ---------


@pytest.fixture
def fail_after_seconds():
    """Turn a hang into a test failure instead of blocking the suite."""
    import signal

    def _timeout(*_):
        raise TimeoutError("chunking did not terminate")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(10)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(("size", "overlap"), [(10, 2), (10, 9), (10, 10), (10, 25)])
def test_fixed_size_with_overlap_terminates_and_covers_text(fail_after_seconds, size, overlap):
    chunker = SemanticChunker(
        strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=size, chunk_overlap=overlap
    )
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunker.chunk(text, "doc-1")
    assert chunks
    assert chunks[0].text.startswith("word0")
    assert chunks[-1].text.rstrip().endswith("word199")


def test_default_chunker_handles_long_unbroken_token(fail_after_seconds):
    # A long URL or inline base64 blob has no separators to split on.
    blob = "aGVsbG8" * 3000
    chunks = SemanticChunker().chunk(blob, "doc-1")
    assert chunks
    assert "".join(c.text for c in chunks).replace(" ", "").startswith(blob[:100])
