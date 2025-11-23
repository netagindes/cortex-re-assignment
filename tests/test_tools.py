import pytest

from app import tools
from app.data_layer import load_assets


def test_format_currency():
    assert tools.format_currency(1234.56) == "USD 1,234.56"

