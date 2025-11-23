import pytest

from app import tools
from app.data_layer import load_assets


def _load_sample_frame():
    df = load_assets()
    sample = df[
        (df.get("entity") == "PropCo")
        & (df.get("property_name") == "Building 180")
        & (df.get("tenant_name") == "Tenant 14")
        & (df.get("quarter") == "2025-Q1")
    ]
    if sample.empty:
        pytest.skip("Dataset no longer contains the Building 180 / Tenant 14 Q1 2025 slice.")
    return sample


def test_monthly_noi_for_building_180_q1():
    sample = _load_sample_frame()
    agg = tools.aggregate_pnl(sample, period_level="month")
    totals = agg["totals"]

    def noi(month_code: str) -> float:
        column = totals.loc[totals["month"] == month_code, "net_operating_income"]
        assert not column.empty, f"No totals found for {month_code}"
        return float(column.iloc[0])

    m01 = round(noi("2025-M01"), 2)
    m02 = round(noi("2025-M02"), 2)
    m03 = round(noi("2025-M03"), 2)

    assert m01 == pytest.approx(27100.59, rel=0, abs=0.01)
    assert m02 == pytest.approx(27100.59, rel=0, abs=0.01)
    assert m03 == pytest.approx(27100.60, rel=0, abs=0.01)
    assert round(m01 + m02 + m03, 2) == pytest.approx(81301.78, rel=0, abs=0.01)


@pytest.mark.skip(reason="Add parking-only and discount-only aggregation regression tests once derived metrics land.")
def test_parking_income_slice_placeholder():
    """Documented placeholder for future derived-metric regression tests."""
    raise NotImplementedError

