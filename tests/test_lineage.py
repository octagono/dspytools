"""Test program lineage (Feature #3)."""

from dspytools.core.registry import compute_dataset_hash


def test_dataset_hash_deterministic():
    data = [{"input": "hello", "output": "world"}, {"input": "foo", "output": "bar"}]
    h1 = compute_dataset_hash(data)
    h2 = compute_dataset_hash(data)
    assert h1 == h2
    assert len(h1) == 12


def test_dataset_hash_different_data():
    h1 = compute_dataset_hash([{"a": 1}])
    h2 = compute_dataset_hash([{"a": 2}])
    assert h1 != h2


def test_dataset_hash_empty():
    h = compute_dataset_hash([])
    assert len(h) == 12
