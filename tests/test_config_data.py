import numpy as np
import pytest

from fed_skdpp.config import (
    DataConfig, ExperimentConfig, KernelConfig,
    SearchConfig, SelectionConfig, load_config,
)
from fed_skdpp.data import decode_kmers, partition_indices, sample_kmers


def test_default_config_matches_requested_sizes():
    config = load_config("config.yml")
    assert config.data.kmer_length == 6
    assert config.data.global_collection_size == 1000
    assert config.partition.num_clients == 10
    assert config.client_collection_size == 100
    assert config.selection.client_selection_size == 50
    assert config.selection.forced_set_size == 10
    assert config.selection.final_selection_size == 100
    assert config.server_pool_size == 500
    assert config.selection.final_objective == "pooled_context"
    assert config.search.patience == 50


@pytest.mark.parametrize("kwargs", [
    {"kmer_length": 0}, {"global_collection_size": 4097},
    {"alphabet": "AACG"}, {"alphabet": "A"}, {"kmer_length": True},
])
def test_invalid_data_settings(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


@pytest.mark.parametrize("gamma", [0, -1, float("nan"), float("inf"), True])
def test_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        KernelConfig(gamma=gamma)


def test_cross_field_validation():
    with pytest.raises(ValueError, match="divisible"):
        ExperimentConfig(data=DataConfig(global_collection_size=101))
    with pytest.raises(ValueError, match="client's collection"):
        ExperimentConfig(selection=SelectionConfig(client_selection_size=101))
    with pytest.raises(ValueError, match="server candidate"):
        ExperimentConfig(selection=SelectionConfig(client_selection_size=1, final_selection_size=11))
    with pytest.raises(ValueError, match="MCMC"):
        SearchConfig(algorithm="mcmc")


def test_config_rejects_typo(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("kernel:\n  gama: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown kernel"):
        load_config(path)


def test_sampling_is_unique_valid_and_reproducible():
    config = DataConfig()
    left = sample_kmers(config, np.random.default_rng(42))
    right = sample_kmers(config, np.random.default_rng(42))
    assert np.array_equal(left, right)
    assert left.shape == (1000, 6)
    assert len(np.unique(left, axis=0)) == 1000
    assert ((left >= 0) & (left < 4)).all()
    strings = decode_kmers(left, config.alphabet)
    assert len(set(strings)) == 1000
    assert all(len(string) == 6 and set(string) <= set("ACGT") for string in strings)


def test_sampling_handles_long_kmers_without_enumeration():
    config = DataConfig(kmer_length=37, global_collection_size=25)
    encoded = sample_kmers(config, np.random.default_rng(8))
    assert encoded.shape == (25, 37)
    assert len(np.unique(encoded, axis=0)) == 25


def test_balanced_partition_has_no_duplicates_or_missing_items():
    parts = partition_indices(1000, 10, np.random.default_rng(2))
    assert all(len(part) == 100 for part in parts)
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(1000))
    assert all(np.array_equal(part, np.sort(part)) for part in parts)
    with pytest.raises(ValueError):
        partition_indices(101, 10, np.random.default_rng(2))
