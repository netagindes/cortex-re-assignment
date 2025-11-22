import pytest

from app import tools
from app.data_layer import load_assets


def test_format_currency():
    assert tools.format_currency(1234.56) == "USD 1,234.56"


def test_pnl_aggregation_sample_data():
    df = load_assets()
    sample = df[
        (df.get("entity") == "PropCo")
        & (df.get("property_name") == "Building 180")
        & (df.get("tenant_name") == "Tenant 14")
        & (df.get("quarter") == "2025-Q1")
    ]
    agg = tools.aggregate_pnl(sample, period_level="month")
    totals = agg["totals"]

    def _net(month: str) -> float:
        match = totals.loc[totals["month"] == month, "net_operating_income"]
        assert not match.empty
        return round(float(match.iloc[0]), 2)

    m01 = _net("2025-M01")
    m02 = _net("2025-M02")
    m03 = _net("2025-M03")

    # These reference values are computed directly from the bundled dataset.
    assert m01 == pytest.approx(27100.59, rel=0, abs=0.01)
    assert m02 == pytest.approx(27100.59, rel=0, abs=0.01)
    assert m03 == pytest.approx(27100.60, rel=0, abs=0.01)
    assert round(m01 + m02 + m03, 2) == pytest.approx(81301.78, rel=0, abs=0.01)

