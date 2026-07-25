"""Tests for MemoryManager — entity extraction, tag extraction, content hashing, dedup."""

from __future__ import annotations

from dspytools.memory.manager import MemoryManager


class TestContentHash:
    """Tests for MemoryManager._content_hash."""

    def test_hash_is_deterministic(self):
        """Same content always produces the same hash."""
        h1 = MemoryManager._content_hash("Hello World")
        h2 = MemoryManager._content_hash("Hello World")
        assert h1 == h2

    def test_hash_normalizes_case_and_whitespace(self):
        """Hash ignores case and leading/trailing whitespace."""
        assert MemoryManager._content_hash("  Hello  ") == MemoryManager._content_hash(
            "hello"
        )
        assert MemoryManager._content_hash("Hello") == MemoryManager._content_hash(
            "HELLO"
        )

    def test_hash_is_16_chars(self):
        """Content hash is truncated to 16 characters."""
        h = MemoryManager._content_hash("test content")
        assert len(h) == 16

    def test_different_content_different_hash(self):
        """Different content produces different hashes."""
        assert MemoryManager._content_hash("apple") != MemoryManager._content_hash(
            "banana"
        )


class TestEntityExtraction:
    """Tests for MemoryManager._extract_entities."""

    def test_extracts_capitalized_words(self):
        """Capitalized words are extracted as entities."""
        entities = MemoryManager._extract_entities("John went to New York")
        names = [e[0] for e in entities]
        assert "John" in names
        assert "New York" in names

    def test_extracts_multi_word_entities(self):
        """Multi-word capitalized sequences are extracted as single entities."""
        entities = MemoryManager._extract_entities("we visited United States last year")
        names = [e[0] for e in entities]
        assert "United States" in names

    def test_all_caps_not_extracted(self):
        """All-caps acronyms (NASA, FBI) are not matched by the Title Case regex.
        This documents the known limitation — regex expects [A-Z][a-z]+."""
        entities = MemoryManager._extract_entities("NASA launched a rocket")
        names = [e[0] for e in entities]
        # NASA is all-caps, doesn't match [A-Z][a-z]+ pattern
        assert "NASA" not in names

    def test_abstract_suffixes(self):
        """Words ending in -tion/-ment/-ity/-ness are typed as 'abstract'."""
        entities = MemoryManager._extract_entities("The Creation was Invention")
        types = {e[0]: e[1] for e in entities}
        # Both should be abstract type
        assert any(v == "abstract" for v in types.values())

    def test_no_duplicates(self):
        """Duplicate entities are not returned twice."""
        entities = MemoryManager._extract_entities("John likes John")
        names = [e[0] for e in entities]
        assert names.count("John") == 1

    def test_empty_content(self):
        """Empty string returns no entities."""
        assert MemoryManager._extract_entities("") == []


class TestTagExtraction:
    """Tests for MemoryManager._extract_tags."""

    def test_extracts_lowercase_words(self):
        """Tags are lowercase words with 4+ characters."""
        tags = MemoryManager._extract_tags("the quick brown fox jumps")
        assert "quick" in tags
        assert "brown" in tags
        assert "jumps" in tags

    def test_filters_short_words(self):
        """Words shorter than 4 characters are excluded."""
        tags = MemoryManager._extract_tags("cat dog run")
        assert tags == []

    def test_filters_stopwords(self):
        """Common stopwords are excluded."""
        tags = MemoryManager._extract_tags("this that with from have been")
        assert tags == []

    def test_deduplicates(self):
        """Duplicate tags are removed."""
        tags = MemoryManager._extract_tags("hello hello world world")
        assert len(tags) == 2

    def test_max_10_tags(self):
        """At most 10 tags are returned."""
        words = " ".join(f"word{i}" for i in range(20))
        tags = MemoryManager._extract_tags(words)
        assert len(tags) <= 10

    def test_empty_content(self):
        """Empty string returns no tags."""
        assert MemoryManager._extract_tags("") == []


class TestMemoryManagerSingleton:
    """Tests for MemoryManager singleton pattern."""

    def test_singleton_identity(self):
        """MemoryManager is a singleton — same instance."""
        m1 = MemoryManager()
        m2 = MemoryManager()
        assert m1 is m2
