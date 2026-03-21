---
description: How to run the Nifty Factor Index Ranking Tool (Alpha 50, Momentum 30, Low Vol 50)
---

# 📊 Nifty Factor Index Ranking Workflow

Implements the **exact NSE Methodology (Feb 2026)** for all major Nifty Factor Indices.

## 🛠 Prerequisites

```bash
pip install pandas numpy yfinance scipy
```

## 🚀 Available Factors

| Factor | NSE Index | Command |
|---|---|---|
| **Alpha** | Nifty Alpha 50, Nifty100 Alpha 30, Nifty200 Alpha 30 | `--factor alpha` |
| **Momentum** | Nifty200 Momentum 30, Nifty500 Momentum 50, Nifty Midcap150 Momentum 50 | `--factor momentum` |
| **Dual Momentum** | Absolute + Relative Momentum combined | `--factor dual_momentum` |
| **Volatility** | Nifty Low Vol 50, Nifty100 Low Vol 30, Nifty500 Low Vol 50 | `--factor volatility` |
| **Beta** | Nifty High Beta 50 | `--factor beta` |
| **Quality** | Nifty100 Quality 30, Nifty500 Quality 50 (ROE, D/E, EPS variability) | `--factor quality` |
| **Value** | E/P, B/P, S/P, Dividend Yield | `--factor value` |
| **Multifactor** | Nifty500 MQVLv 50 (25% each: Momentum + Quality + Value + Low Vol) | `--factor multifactor` |

## 📋 Example Commands

### Alpha 50 (Nifty 500 universe)
// turbo
```bash
python run_factor_ranking.py --factor alpha --index "NIFTY 500" --symbols 50
```

### Dual Momentum 30 (Nifty 200 universe)
// turbo
```bash
python run_factor_ranking.py --factor dual_momentum --index "NIFTY 200" --symbols 30
```

### Momentum 50 (Midcap universe)
// turbo
```bash
python run_factor_ranking.py --factor momentum --index "NIFTY MIDCAP 100" --symbols 50
```

### Low Volatility 50 (Nifty 500 universe)
// turbo
```bash
python run_factor_ranking.py --factor volatility --index "NIFTY 500" --symbols 50
```

### High Beta 50 (Nifty 500 universe)
// turbo
```bash
python run_factor_ranking.py --factor beta --index "NIFTY 500" --symbols 50
```

### Quality 30 (Nifty 100 universe — fetches fundamentals)
```bash
python run_factor_ranking.py --factor quality --index "NIFTY 100" --symbols 30
```

### Multifactor MQVLv 50 (full composite — slow, fetches fundamentals)
```bash
python run_factor_ranking.py --factor multifactor --index "NIFTY 500" --symbols 50
```

## 📁 Output

Reports saved to `data/reports/rankings/` as CSVs.

## 🧮 Methodology Summary

| Factor | Calculation | Source |
|---|---|---|
| **Alpha** | Jensen's Alpha = annualized intercept of regression vs Nifty 50 | 1-year trailing prices |
| **Beta** | Slope of regression of stock returns vs Nifty 50 returns | 1-year trailing prices |
| **Momentum** | `MR = Price Return / Annualized Vol`; Z-score 50% MR12 + 50% MR6; Normalized | 6m & 12m returns |
| **Dual Momentum** | `Absolute (is 12m return > 0?) × Relative (Momentum Score)` | Combined |
| **Low Volatility** | Inverse of annualized SD of log-normal returns; Z-score normalized | 1-year trailing prices |
| **Quality** | `1/3 Z(ROE) + 1/3 -Z(D/E) + 1/3 -Z(EPS variability)` (financials: 50/50 ROE/EPS) | Fiscal year data |
| **Value** | `0.25 Z(E/P) + 0.25 Z(B/P) + 0.25 Z(S/P) + 0.25 Z(DivYield)` | Fiscal year data |
| **Multifactor** | `25% Momentum + 25% Quality + 25% Value + 25% LowVol` (composite + percentile) | All above |

## 🤖 Automation

GitHub Action at `.github/workflows/factor_rankings.yml` runs monthly on the 1st.
