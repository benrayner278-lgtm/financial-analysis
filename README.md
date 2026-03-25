# Financial Market Analysis

A quantitative analysis of multi-asset portfolio performance using Python.
Computes return metrics, risk measures, rolling statistics, and simulates
the efficient frontier via Monte Carlo methods.

Built as a demonstration of quantitative finance fundamentals — the same
concepts used in systematic trading and portfolio research.

---

## What It Does

### 1. Data Pipeline
Downloads five years of historical price data for a portfolio of assets
across equities, commodities, fixed income, and crypto. Falls back to
simulated data generated via correlated Geometric Brownian Motion if
offline, so the project runs in any environment.

**Assets analysed:**
| Ticker | Asset |
|--------|-------|
| SPY | S&P 500 ETF |
| QQQ | NASDAQ-100 ETF |
| GLD | Gold ETF |
| TLT | Long-Term Bond ETF |
| BTC-USD | Bitcoin |

---

### 2. Risk & Return Metrics

For each asset the project computes:

| Metric | Description |
|--------|-------------|
| **Annualised Return** | Compound annual growth rate from daily log returns |
| **Annualised Volatility** | Standard deviation of returns scaled to annual |
| **Sharpe Ratio** | Risk-adjusted return above the risk-free rate |
| **Maximum Drawdown** | Worst peak-to-trough decline over the period |
| **Value at Risk (95%)** | Worst expected daily loss 95% of the time |
| **Expected Shortfall (CVaR)** | Average loss in the worst 5% of days |

---

### 3. Visualisations

Six charts are generated automatically into `/outputs/`:

**01 — Normalised Price Performance**
All assets rebased to 100 for direct comparison regardless of starting price.

**02 — Return Distributions**
Daily return histograms with 95% Value at Risk overlaid — shows fat tails
and asymmetry in financial returns (non-normality).

**03 — Correlation Heatmap**
Full correlation matrix across all assets. Key finding: equities and
bonds tend to have negative correlation — the basis of diversification.

**04 — Rolling Sharpe Ratio**
252-day rolling Sharpe ratio over time. Shows how risk-adjusted performance
varies across market regimes — a technique used in systematic strategy
monitoring.

**05 — Drawdown Chart**
Underwater equity curves showing time spent below all-time highs. Useful
for understanding recovery time and psychological difficulty of strategies.

**06 — Monte Carlo Efficient Frontier**
3,000 randomly weighted portfolios plotted in risk-return space. The
maximum Sharpe portfolio is highlighted — demonstrates the core insight
of Modern Portfolio Theory: diversification improves risk-adjusted returns.

---

### 4. Key Findings

Running the analysis on live data reveals several interesting patterns:

- **Equities dominate returns** but with significantly higher drawdowns
- **Gold and bonds provide genuine diversification** — negative or near-zero
  correlation with equities reduces portfolio volatility substantially
- **Bitcoin's Sharpe ratio** is highly variable — exceptional in bull markets,
  deeply negative in bear markets
- **Rolling Sharpe charts** show clear regime changes — the 2022 rate-hiking
  cycle caused simultaneous drawdowns in both equities and bonds, temporarily
  breaking the traditional negative correlation
- **The efficient frontier** confirms that no single asset dominates —
  optimal portfolios always blend multiple assets

---

## Technical Approach

### Why Log Returns?
Log returns are used rather than simple returns because they are:
- **Additive across time** — daily log returns sum to give period return
- **Normally distributed** (approximately) — enables statistical analysis
- **Consistent with GBM** — the standard model for asset price dynamics

### Correlated Brownian Motion (Simulated Data)
When offline, prices are generated using Geometric Brownian Motion with a
Cholesky decomposition of the correlation matrix. This ensures simulated
assets have realistic co-movement — SPY and QQQ correlate highly, gold
acts as a defensive asset, bonds hedge equity risk.

### Monte Carlo Portfolio Simulation
3,000 portfolios are generated using Dirichlet-distributed random weights
(which naturally sum to 1 and are always positive). Each portfolio's
annualised return, volatility and Sharpe ratio are computed and plotted,
tracing out the efficient frontier empirically rather than analytically.

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/benrayner278-lgtm/financial-analysis.git
cd financial-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python src/analysis.py
```

Outputs are saved to `/outputs/`. The script works offline using simulated
data if `yfinance` cannot connect.

---

## Project Structure

```
financial-analysis/
├── src/
│   └── analysis.py        # Main analysis script
├── outputs/               # Generated charts and CSV (git-ignored)
├── requirements.txt
└── README.md
```

---

## Extensions (Planned)

- [ ] Add a simple momentum signal and backtest it
- [ ] Implement minimum variance and equal-risk-contribution portfolios
- [ ] Add regime detection using Hidden Markov Models
- [ ] Compare empirical return distributions to fitted normal and t-distributions
- [ ] Add rolling beta calculation against SPY

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and time series handling |
| `numpy` | Numerical computation |
| `matplotlib` | Visualisation |
| `yfinance` | Market data download |
| `scipy` | Statistical functions |

---

## Background

Built as a self-directed project to deepen understanding of quantitative
finance fundamentals while studying Mathematics and Physics at the
University of Manchester. The concepts here — returns, risk metrics,
correlation, and portfolio optimisation — form the mathematical
foundation of systematic trading and quantitative research.

---

*Benjamin Rayner — github.com/benrayner278-lgtm*