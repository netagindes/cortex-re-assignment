import json

import pytest

from app import tools
from app.data_layer import load_assets
from app.knowledge import get_property_memory
from app.knowledge import property_memory as property_memory_module


def _two_sample_addresses() -> list[str]:
    df = load_assets()
    addresses = df["address"].dropna().unique().tolist()
    if len(addresses) < 2:
        pytest.skip("Dataset does not contain enough addresses for this test")
    return addresses[:2]


def test_property_memory_search_returns_known_address():
    memory = get_property_memory()
    samples = _two_sample_addresses()
    results = memory.search(samples[0], top_k=1)
    assert results, "PropertyMemory returned no matches"
    assert results[0].address == samples[0]
    assert results[0].confidence > 0


def test_property_memory_resolves_multiple_mentions():
    memory = get_property_memory()
    first, second = _two_sample_addresses()
    query = f"Compare {first} vs {second} for me"
    result = memory.resolve_mentions(query, expected=2)
    assert len(result.matches) == 2
    resolved_addresses = [match.address for match in result.matches]
    assert first in resolved_addresses and second in resolved_addresses
    assert not result.unresolved_terms


def test_property_resolution_uses_aliases(tmp_path, monkeypatch):
    df = load_assets()
    addresses = df["address"].dropna().unique().tolist()
    if len(addresses) < 1:
        pytest.skip("Dataset does not provide any addresses")

    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(json.dumps({"123 main st": addresses[0]}))

    monkeypatch.setenv("ADDRESS_ALIAS_FILE", str(alias_file))
    tools._address_aliases.cache_clear()
    property_memory_module.get_property_memory.cache_clear()

    result = tools.resolve_properties("Compare 123 Main St with another place", max_matches=1)
    tools._address_aliases.cache_clear()
    property_memory_module.get_property_memory.cache_clear()

    assert result.matches
    assert result.matches[0].address == addresses[0]


