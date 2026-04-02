import pandas as pd

from src.screening.extended_signals import score_extended_signals
from src.screening.indicators import calculate_rsi, detect_higher_hl


def test_rsi_all_gains_converges_to_100():
    prices = pd.Series([100 + i for i in range(40)])
    rsi = calculate_rsi(prices, period=14)

    assert rsi.dropna().iloc[-1] == 100.0


def test_detect_higher_hl_identifies_lower_high_lower_low():
    df = pd.DataFrame(
        {
            "High": [110, 109, 108, 107, 106, 103],
            "Low": [100, 99, 98, 97, 96, 90],
            "Close": [105, 104, 103, 102, 101, 92],
            "Open": [104, 103, 102, 101, 100, 94],
            "Volume": [1000] * 6,
        }
    )

    assert detect_higher_hl(df, window=5) == "lower_hl"


def test_extended_signals_emits_macd_signal_on_bullish_cross():
    closes = [100] * 220 + [99, 98, 97, 98, 100, 103, 107, 112, 118, 125]
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [100000] * len(closes),
        }
    )

    result = score_extended_signals("TEST.NS", df)

    assert any("MACD" in signal for signal in result["signals"])
