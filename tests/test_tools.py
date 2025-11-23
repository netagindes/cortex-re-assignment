from types import SimpleNamespace

import pytest

from app import tools
from app.data_layer import load_assets
from app.knowledge import PropertyMatch


def test_format_currency():
    assert tools.format_currency(1234.56) == "USD 1,234.56"


def test_resolve_properties_includes_memory_matches_when_alias_missing(monkeypatch):
    memory_match = PropertyMatch(
        address="Building 180",
        property_name="Building 180",
        confidence=0.75,
        rank=1,
        reason="memory",
        metadata={},
    )

    class FakeMemory:
        def resolve_mentions(self, text: str, expected: int):
            return SimpleNamespace(
                candidate_terms=["Building 180"],
                unresolved_terms=[],
                matches=[memory_match],
            )

    def fake_lookup(_: str, df=None):
        raise ValueError("not found")

    monkeypatch.setattr(tools, "get_property_memory", lambda: FakeMemory())
    monkeypatch.setattr(tools, "extract_addresses", lambda text, max_matches=2: ["Unknown Building"])
    monkeypatch.setattr(tools, "_lookup_asset", fake_lookup)

    resolution = tools.resolve_properties("Unknown Building", max_matches=2)

    addresses = [match.address for match in resolution.matches]
    assert "Building 180" in addresses, "Memory matches should be preserved even if alias lookup fails."


def test_resolve_properties_keeps_memory_matches_when_missing_exceeds_limit(monkeypatch):
    memory_matches = [
        PropertyMatch(
            address="Building 150",
            property_name="Building 150",
            confidence=0.6,
            rank=1,
            reason="memory",
            metadata={},
        )
    ]

    class FakeMemory:
        def resolve_mentions(self, text: str, expected: int):
            return SimpleNamespace(
                candidate_terms=["Building 150"],
                unresolved_terms=[],
                matches=memory_matches,
            )

    def fake_lookup(_: str, df=None):
        raise ValueError("not found")

    monkeypatch.setattr(tools, "get_property_memory", lambda: FakeMemory())
    monkeypatch.setattr(
        tools,
        "extract_addresses",
        lambda text, max_matches=2: ["Unknown A", "Unknown B"],
    )
    monkeypatch.setattr(tools, "_lookup_asset", fake_lookup)

    resolution = tools.resolve_properties("Compare Unknown A and Unknown B", max_matches=1)

    addresses = [match.address for match in resolution.matches]
    assert "Building 150" in addresses, "Semantic matches should remain even when multiple aliases are missing."
