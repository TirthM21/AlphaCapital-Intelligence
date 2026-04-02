import pandas as pd

from backtest_regime_strategies import adjust_returns_for_direction
from src.screening.extended_signals import score_extended_signals
from src.screening.indicators import detect_volume_spike
from src.screening.piped_scanners import run_piped_scan
from src.screening.regime_strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    ShortDefensiveStrategy,
    VolatilityExpansionStrategy,
)


def make_base_df(length: int = 260) -> pd.DataFrame:
    close = pd.Series([100 + i * 0.2 for i in range(length)], dtype=float)
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.3
    volume = pd.Series([1000.0] * length)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def make_momentum_breakout_df(length: int = 260) -> pd.DataFrame:
    rows = []
    price = 100.0
    for i in range(length):
        if i < 220:
            price += 0.22
        elif i < 255:
            price += (-1) ** i * 0.05
        else:
            price += 1.2
        close = price
        open_ = close - 0.4
        high = close + 0.8
        low = close - 0.9
        volume = 1000.0 if i < 255 else 2200.0
        rows.append((open_, high, low, close, volume))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])


def make_mean_reversion_bounce_df(length: int = 260) -> pd.DataFrame:
    closes = [120.0 + i * 0.18 for i in range(length - 8)]
    closes.extend([165.0, 158.0, 150.0, 143.0, 138.0, 139.5, 144.5, 151.5])
    rows = []
    for i, close in enumerate(closes):
        if i == len(closes) - 2:
            open_ = close - 3.8
            high = close + 1.0
            low = close - 5.5
            volume = 1550.0
        elif i == len(closes) - 1:
            open_ = close - 3.0
            high = close + 2.8
            low = close - 9.5
            volume = 1800.0
        else:
            open_ = close - 0.6
            high = close + 1.0
            low = close - 1.2
            volume = 950.0
        rows.append((open_, high, low, close, volume))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])


def make_short_breakdown_df(length: int = 260) -> pd.DataFrame:
    rows = []
    price = 220.0
    for i in range(length):
        if i < 210:
            price -= 0.20
        else:
            price -= 0.35
        close = price
        open_ = close + 0.6
        high = close + 1.0
        low = close - 1.1
        volume = 1000.0 if i < 250 else 1500.0
        rows.append((open_, high, low, close, volume))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])


def make_volatility_expansion_df(length: int = 260) -> pd.DataFrame:
    rows = []
    price = 90.0
    for i in range(length):
        if i < 220:
            price += 0.12
            spread = 1.8
        elif i < 255:
            price += 0.02
            spread = 0.5
        else:
            price += 0.95
            spread = 2.1
        close = price
        open_ = close - 0.3
        high = close + spread
        low = close - spread
        volume = 1000.0 if i < 255 else 2300.0
        rows.append((open_, high, low, close, volume))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])


def test_detect_volume_spike_uses_prior_bars_only():
    volumes = pd.Series([100.0] * 20 + [210.0])
    assert detect_volume_spike(volumes, volumes.iloc[-1], threshold=2.0)


def test_52_week_breakout_requires_new_high():
    df = make_base_df()
    prior_high = df["High"].iloc[-2]
    df.loc[df.index[-1], "Close"] = prior_high
    df.loc[df.index[-1], "High"] = prior_high

    result = score_extended_signals("TEST.NS", df)

    assert "52-Week High Breakout" not in result["signals"]


def test_pipe_1_does_not_trigger_on_rsi_overbought_alone():
    df = make_base_df()
    base = {
        "ticker": "TEST.NS",
        "signals": [
            "Volume Breakout (>2x)",
            "RSI Overbought",
            "52-Week High Breakout",
        ],
        "price": 150.0,
        "rsi": 78.0,
        "prediction_score": 70,
        "lorentzian": 60.0,
        "bullish_score": 1,
        "bearish_score": 2,
    }

    # simulate run_piped_scan using a monkeypatched base scorer by wrapping current result
    import src.screening.piped_scanners as piped

    original = piped.score_extended_signals
    piped.score_extended_signals = lambda ticker, hist_df: dict(base)
    try:
        result = run_piped_scan("TEST.NS", df, 1)
    finally:
        piped.score_extended_signals = original

    assert not result["setup_found"]


def test_pipe_4_requires_bullish_reversal_not_any_candlestick():
    df = make_base_df()
    base = {
        "ticker": "TEST.NS",
        "signals": [
            "Bullish RSI Divergence",
            "Candlestick: Shooting Star",
        ],
        "price": 150.0,
        "rsi": 28.0,
        "prediction_score": 68,
        "lorentzian": 55.0,
        "bullish_score": 2,
        "bearish_score": 2,
    }

    import src.screening.piped_scanners as piped

    original = piped.score_extended_signals
    piped.score_extended_signals = lambda ticker, hist_df: dict(base)
    try:
        result = run_piped_scan("TEST.NS", df, 4)
    finally:
        piped.score_extended_signals = original

    assert not result["setup_found"]


def test_momentum_strategy_requires_real_breakout_quality():
    weak_df = make_base_df()
    strong_df = make_momentum_breakout_df()

    weak = MomentumStrategy("TEST.NS", weak_df).generate_signal()
    strong = MomentumStrategy("TEST.NS", strong_df).generate_signal()

    assert weak["signal_strength"] == 0.0
    assert strong["signal_strength"] > 0.0


def test_mean_reversion_strategy_requires_actual_bounce():
    weak_df = make_base_df()
    strong_df = make_mean_reversion_bounce_df()

    weak = MeanReversionStrategy("TEST.NS", weak_df).generate_signal()
    strong = MeanReversionStrategy("TEST.NS", strong_df).generate_signal()

    assert weak["signal_strength"] == 0.0
    assert strong["signal_strength"] > 0.0


def test_short_strategy_needs_real_downtrend_breakdown():
    weak_df = make_base_df()
    strong_df = make_short_breakdown_df()

    weak = ShortDefensiveStrategy("TEST.NS", weak_df).generate_signal()
    strong = ShortDefensiveStrategy("TEST.NS", strong_df).generate_signal()

    assert weak["signal_strength"] == 0.0
    assert strong["signal_strength"] > 0.0


def test_volatility_expansion_requires_contraction_then_breakout():
    weak_df = make_base_df()
    strong_df = make_volatility_expansion_df()

    weak = VolatilityExpansionStrategy("TEST.NS", weak_df).generate_signal()
    strong = VolatilityExpansionStrategy("TEST.NS", strong_df).generate_signal()

    assert weak["signal_strength"] == 0.0
    assert strong["signal_strength"] > 0.0


def test_short_strategy_returns_are_inverted():
    perf = {"1D": -2.5, "5D": 4.0, "10D": None}
    adjusted = adjust_returns_for_direction("Short / Defensive", perf)

    assert adjusted == {"1D": 2.5, "5D": -4.0, "10D": None}
