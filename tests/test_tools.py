from app import tools


def test_format_currency():
    assert tools.format_currency(1234.56) == "USD 1,234.56"

