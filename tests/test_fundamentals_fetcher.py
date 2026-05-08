"""Regression tests for fundamental snapshot and analysis helpers."""

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip('pandas')
pytest.importorskip('yfinance')

_SPEC = importlib.util.spec_from_file_location(
    'fundamentals_fetcher_under_test',
    Path(__file__).resolve().parents[1] / 'src' / 'data' / 'fundamentals_fetcher.py',
)
fundamentals_fetcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fundamentals_fetcher)

analyze_fundamentals_for_signal = fundamentals_fetcher.analyze_fundamentals_for_signal
create_fundamental_snapshot = fundamentals_fetcher.create_fundamental_snapshot


def test_create_fundamental_snapshot_handles_explicit_none_values():
    """Missing yfinance metrics represented as None should not crash snapshots."""
    quarterly_data = {
        'revenue_yoy_change': None,
        'revenue_qoq_change': None,
        'eps_yoy_change': None,
        'eps_qoq_change': None,
        'gross_margin': None,
        'margin_change': None,
        'inventory_qoq_change': None,
        'inventory_to_sales_ratio': None,
        'inventory_breakdown_available': False,
    }

    snapshot = create_fundamental_snapshot('ATGL.NS', quarterly_data)

    assert 'FUNDAMENTAL SNAPSHOT - ATGL.NS' in snapshot
    assert 'Revenue: Data not available' in snapshot
    assert 'EPS: Data not available' in snapshot
    assert 'Fundamentals SUPPORT technical breakout' in snapshot


def test_analyze_fundamentals_for_signal_handles_explicit_none_values():
    """Explicit None metrics should be treated like missing neutral data."""
    analysis = analyze_fundamentals_for_signal({
        'revenue_yoy_change': None,
        'revenue_qoq_change': None,
        'eps_yoy_change': None,
        'inventory_qoq_change': None,
    })

    assert analysis['revenue_trend'] == 'flat'
    assert analysis['eps_trend'] == 'flat'
    assert analysis['inventory_signal'] == 'neutral'
    assert analysis['sequential_revenue_declining'] is False
    assert analysis['penalty_points'] == 0
