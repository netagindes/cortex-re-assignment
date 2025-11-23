import pytest

from app.classifier import ClassificationLayer


@pytest.fixture(scope="module")
def classifier() -> ClassificationLayer:
    return ClassificationLayer(enable_llm=False)


def test_classifier_identifies_pnl_with_property_and_period(classifier: ClassificationLayer) -> None:
    result = classifier.classify("Show me the P&L for Building 180 for March 2025.")
    assert result.request_type == "pnl"
    assert "Building 180" in result.addresses
    assert result.period == "2025-M03"
    assert not result.missing_fields


def test_classifier_marks_missing_period_for_pnl(classifier: ClassificationLayer) -> None:
    result = classifier.classify("Give me the P&L for Building 180.")
    assert result.request_type == "pnl"
    assert "property" not in result.missing_fields
    assert "period" in result.missing_fields


def test_classifier_handles_price_comparison(classifier: ClassificationLayer) -> None:
    result = classifier.classify("What is the price of 123 Main St vs 456 Oak Ave?")
    assert result.request_type == "price_comparison"
    assert "123 Main St" in result.addresses
    assert "456 Oak Ave" in result.addresses


def test_classifier_handles_asset_details(classifier: ClassificationLayer) -> None:
    result = classifier.classify("Tell me about Building 220.")
    assert result.request_type == "asset_details"
    assert "Building 220" in result.addresses
    assert not result.missing_fields


def test_classifier_general_definition(classifier: ClassificationLayer) -> None:
    result = classifier.classify("What is NOI?")
    assert result.request_type == "general"
    assert not result.addresses
    assert not result.period


def test_classifier_detects_period_comparison(classifier: ClassificationLayer) -> None:
    result = classifier.classify("Compare January 2025 and February 2025 P&L for Building 180.")
    assert result.request_type == "pnl"
    assert result.comparison_periods == ["2025-M01", "2025-M02"]

