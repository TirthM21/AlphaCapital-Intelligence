from run_comprehensive_longterm_backtest import (
    parse_factor_summary,
    parse_signal_summary,
    parse_vcp_summary,
)


def test_parse_factor_summary_extracts_rows():
    output = """
      FACTOR STRATEGY BACKTEST COMPARISON
      🟢 momentum          +12.50%   +5.25%   +7.25%   66.7%   +8.10%  +21.00%   -4.50%    30
      🔴 value             -1.50%   +5.25%   -6.75%   40.0%   -2.40%   +6.80%   -9.10%    30
    """

    rows = parse_factor_summary(output)

    assert len(rows) == 2
    assert rows[0]["factor"] == "momentum"
    assert rows[0]["alpha"] == 7.25
    assert rows[1]["factor"] == "value"
    assert rows[1]["stocks"] == 30


def test_parse_signal_summary_extracts_count_and_periods():
    output = """
================================================================================
  ALL EXTENDED SIGNALS (15 signals)
================================================================================
    1D: Avg   +1.25%  Med   +1.10%  WinRate  60.0%  Best   +6.40%  Worst   -3.20%
   22D: Avg   +4.50%  Med   +3.80%  WinRate  73.3%  Best  +14.00%  Worst   -5.00%
"""

    summary = parse_signal_summary(output, "ALL EXTENDED SIGNALS")

    assert summary["count"] == 15
    assert summary["periods"]["1D"]["avg"] == 1.25
    assert summary["periods"]["22D"]["win_rate"] == 73.3


def test_parse_vcp_summary_extracts_scan_counts_and_periods():
    output = """
  Scanned: 500 stocks | Signals found: 18

      1D       +0.80%    +0.50%     55.6%    +4.20%    -2.10%
     30D       +7.40%    +6.10%     72.2%   +18.00%    -4.50%
"""

    summary = parse_vcp_summary(output)

    assert summary["scanned"] == 500
    assert summary["signals"] == 18
    assert summary["periods"]["30D"]["best"] == 18.0
