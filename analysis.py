"""
Financial Market Analysis
=========================
Analyses historical price data for a portfolio of assets, computing
returns, risk metrics, and correlations. Designed as a demonstration
of quantitative finance fundamentals using Python.

Author: Benjamin Rayner
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

TICKERS = {
    'SPY':  'S&P 500 ETF',
    'QQQ':  'NASDAQ-100 ETF',
    'GLD':  'Gold ETF',
    'TLT':  'Long-Term Bonds ETF',
    'BTC-USD': 'Bitcoin'
}

PERIOD = '5y'           # 5 years of data
RISK_FREE_RATE = 0.045  # Approximate current UK/US risk-free rate (annualised)
TRADING_DAYS = 252      # Standard assumption

# Colour palette — clean dark theme
COLORS = {
    'SPY':    '#4A7CF0',
    'QQQ':    '#C9A84C',
    'GLD':    '#F0A04A',
    'TLT':    '#4AADE8',
    'BTC-USD':'#F07070',
    'bg':     '#0A0B0E',
    'surface':'#0F1117',
    'text':   '#ECF0F5',
    'muted':  '#6B7280',
    'border': '#1F2937',
    'green':  '#4ADE80',
    'red':    '#F87171',
}


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_data(tickers: dict, period: str) -> pd.DataFrame:
    """
    Download historical adjusted close prices.
    Falls back to simulated data if yfinance is unavailable,
    so the project works offline or in any environment.
    """
    try:
        import yfinance as yf
        print("Downloading price data via yfinance...")
        raw = yf.download(
            list(tickers.keys()),
            period=period,
            progress=False,
            auto_adjust=True
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']]
        prices = prices.dropna(how='all')
        print(f"  Downloaded {len(prices)} trading days of data.\n")
        return prices

    except Exception as e:
        print(f"yfinance unavailable ({e}). Generating simulated data...\n")
        return _simulate_prices(tickers, period)


def _simulate_prices(tickers: dict, period: str) -> pd.DataFrame:
    """
    Generate realistic simulated price paths using geometric Brownian motion.
    Annual return / volatility assumptions are based on long-run historical averages.
    """
    np.random.seed(42)
    n_years = int(period.replace('y', ''))
    n_days = n_years * TRADING_DAYS

    dates = pd.bdate_range(
        end=pd.Timestamp.today(),
        periods=n_days
    )

    # (annualised_mu, annualised_sigma, start_price)
    params = {
        'SPY':     (0.10, 0.16, 300.0),
        'QQQ':     (0.14, 0.22, 250.0),
        'GLD':     (0.06, 0.14, 160.0),
        'TLT':     (0.02, 0.12, 140.0),
        'BTC-USD': (0.60, 0.80, 20000.0),
    }

    # Build a simple correlation structure
    corr_matrix = np.array([
        [1.00,  0.90, -0.10, -0.30,  0.20],
        [0.90,  1.00, -0.15, -0.35,  0.25],
        [-0.10,-0.15,  1.00,  0.20,  0.05],
        [-0.30,-0.35,  0.20,  1.00, -0.10],
        [0.20,  0.25,  0.05, -0.10,  1.00],
    ])
    L = np.linalg.cholesky(corr_matrix)

    dt = 1 / TRADING_DAYS
    n = len(tickers)

    z = np.random.standard_normal((n_days, n))
    z_corr = z @ L.T          # Apply correlation structure

    prices = {}
    for i, ticker in enumerate(tickers.keys()):
        mu, sigma, s0 = params[ticker]
        drift = (mu - 0.5 * sigma ** 2) * dt
        shock = sigma * np.sqrt(dt) * z_corr[:, i]
        log_returns = drift + shock
        price_path = s0 * np.exp(np.cumsum(log_returns))
        prices[ticker] = price_path

    return pd.DataFrame(prices, index=dates)


# ── RETURNS & METRICS ──────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def annualised_return(returns: pd.Series) -> float:
    """Compound annualised growth rate from daily log returns."""
    return np.exp(returns.mean() * TRADING_DAYS) - 1


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily log returns."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe ratio."""
    ann_ret = annualised_return(returns)
    ann_vol = annualised_volatility(returns)
    return (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0


def max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough decline."""
    cumulative = prices / prices.cummax()
    return (cumulative.min() - 1)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR — worst daily loss at given confidence level."""
    return np.percentile(returns, (1 - confidence) * 100)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR) — average loss beyond VaR threshold."""
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


def build_metrics_table(prices: pd.DataFrame,
                         returns: pd.DataFrame,
                         tickers: dict) -> pd.DataFrame:
    """Compile a summary metrics table for all assets."""
    rows = []
    for ticker in prices.columns:
        ret = returns[ticker].dropna()
        rows.append({
            'Asset':               tickers.get(ticker, ticker),
            'Ticker':              ticker,
            'Ann. Return':         annualised_return(ret),
            'Ann. Volatility':     annualised_volatility(ret),
            'Sharpe Ratio':        sharpe_ratio(ret),
            'Max Drawdown':        max_drawdown(prices[ticker]),
            'VaR (95%)':           value_at_risk(ret),
            'CVaR (95%)':          expected_shortfall(ret),
        })
    return pd.DataFrame(rows).set_index('Ticker')


# ── ROLLING METRICS ────────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.DataFrame,
                   window: int = 252) -> pd.DataFrame:
    """Rolling annualised Sharpe ratio over a given window."""
    roll_mean = returns.rolling(window).mean() * TRADING_DAYS
    roll_std  = returns.rolling(window).std()  * np.sqrt(TRADING_DAYS)
    return (roll_mean - RISK_FREE_RATE) / roll_std


def rolling_correlation(returns: pd.DataFrame,
                         asset_a: str,
                         asset_b: str,
                         window: int = 60) -> pd.Series:
    """Rolling pairwise correlation between two assets."""
    return returns[asset_a].rolling(window).corr(returns[asset_b])


# ── PORTFOLIO SIMULATION ───────────────────────────────────────────────────────

def simulate_portfolios(returns: pd.DataFrame,
                         n_portfolios: int = 3000) -> pd.DataFrame:
    """
    Monte Carlo simulation of random portfolio weights.
    Returns a DataFrame of (volatility, return, Sharpe) for each portfolio,
    used to plot the efficient frontier scatter.
    """
    n_assets = len(returns.columns)
    results = []

    for _ in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        port_returns = returns.dot(weights)
        p_ret = annualised_return(port_returns)
        p_vol = annualised_volatility(port_returns)
        p_sharpe = (p_ret - RISK_FREE_RATE) / p_vol if p_vol > 0 else 0
        results.append({
            'Return':     p_ret,
            'Volatility': p_vol,
            'Sharpe':     p_sharpe,
            'Weights':    weights
        })

    return pd.DataFrame(results)


# ── VISUALISATION ──────────────────────────────────────────────────────────────

def apply_dark_style():
    """Apply a consistent dark theme to all matplotlib figures."""
    plt.rcParams.update({
        'figure.facecolor':  COLORS['bg'],
        'axes.facecolor':    COLORS['surface'],
        'axes.edgecolor':    COLORS['border'],
        'axes.labelcolor':   COLORS['muted'],
        'axes.titlecolor':   COLORS['text'],
        'xtick.color':       COLORS['muted'],
        'ytick.color':       COLORS['muted'],
        'grid.color':        COLORS['border'],
        'grid.alpha':        0.5,
        'text.color':        COLORS['text'],
        'legend.facecolor':  COLORS['surface'],
        'legend.edgecolor':  COLORS['border'],
        'figure.dpi':        150,
        'font.family':       'monospace',
        'font.size':         9,
        'axes.titlesize':    11,
        'axes.labelsize':    9,
    })


def plot_normalised_prices(prices: pd.DataFrame,
                            tickers: dict,
                            out_path: str):
    """Plot all assets rebased to 100 for direct comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    rebased = (prices / prices.iloc[0]) * 100

    for ticker in rebased.columns:
        color = COLORS.get(ticker, '#FFFFFF')
        label = tickers.get(ticker, ticker)
        ax.plot(rebased.index, rebased[ticker],
                color=color, linewidth=1.5, label=label, alpha=0.9)

    ax.axhline(100, color=COLORS['muted'], linewidth=0.6,
               linestyle='--', alpha=0.4)
    ax.set_title('Normalised Price Performance (Base = 100)', pad=12)
    ax.set_ylabel('Indexed Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {out_path}")


def plot_return_distributions(returns: pd.DataFrame,
                               tickers: dict,
                               out_path: str):
    """Plot return distribution histograms with VaR overlaid."""
    n = len(returns.columns)
    fig, axes = plt.subplots(1, n, figsize=(16, 4), sharey=False)

    for ax, ticker in zip(axes, returns.columns):
        ret = returns[ticker].dropna() * 100   # Express as %
        color = COLORS.get(ticker, '#FFFFFF')
        var = value_at_risk(returns[ticker]) * 100

        ax.hist(ret, bins=80, color=color, alpha=0.7,
                edgecolor='none', density=True)
        ax.axvline(var, color=COLORS['red'], linewidth=1.5,
                   linestyle='--', label=f'VaR 95%: {var:.2f}%')
        ax.axvline(0, color=COLORS['muted'], linewidth=0.8, alpha=0.5)

        ax.set_title(tickers.get(ticker, ticker), fontsize=9)
        ax.set_xlabel('Daily Return (%)')
        ax.legend(fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].set_ylabel('Density')
    fig.suptitle('Return Distributions with Value at Risk (95%)',
                 fontsize=11, y=1.02, color=COLORS['text'])
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {out_path}")


def plot_correlation_heatmap(returns: pd.DataFrame,
                              tickers: dict,
                              out_path: str):
    """Plot the correlation matrix as a colour-coded heatmap."""
    corr = returns.corr()
    labels = [tickers.get(t, t) for t in corr.columns]

    fig, ax = plt.subplots(figsize=(8, 7))
    n = len(corr)
    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            text_col = 'black' if abs(val) < 0.5 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, color=text_col, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title('Asset Correlation Matrix', pad=14)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', color=COLORS['muted'])
    cbar.ax.yaxis.set_tick_params(color=COLORS['muted'])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {out_path}")


def plot_rolling_sharpe(returns: pd.DataFrame,
                         tickers: dict,
                         out_path: str):
    """Plot 252-day rolling Sharpe ratios."""
    roll = rolling_sharpe(returns)
    fig, ax = plt.subplots(figsize=(14, 5))

    for ticker in roll.columns:
        color = COLORS.get(ticker, '#FFFFFF')
        label = tickers.get(ticker, ticker)
        ax.plot(roll.index, roll[ticker],
                color=color, linewidth=1.2, label=label, alpha=0.85)

    ax.axhline(0, color=COLORS['muted'], linewidth=0.8,
               linestyle='--', alpha=0.5)
    ax.axhline(1, color=COLORS['green'], linewidth=0.6,
               linestyle=':', alpha=0.4, label='Sharpe = 1')
    ax.set_title('Rolling 252-Day Sharpe Ratio', pad=12)
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {out_path}")


def plot_efficient_frontier(portfolios: pd.DataFrame,
                             returns: pd.DataFrame,
                             tickers: dict,
                             out_path: str):
    """
    Plot the Monte Carlo efficient frontier scatter with individual
    assets overlaid and the maximum Sharpe portfolio highlighted.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter of random portfolios coloured by Sharpe
    sc = ax.scatter(
        portfolios['Volatility'] * 100,
        portfolios['Return'] * 100,
        c=portfolios['Sharpe'],
        cmap='plasma',
        alpha=0.35,
        s=8,
        linewidths=0
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('Sharpe Ratio', color=COLORS['muted'])

    # Maximum Sharpe portfolio
    max_sharpe_idx = portfolios['Sharpe'].idxmax()
    max_sharpe = portfolios.loc[max_sharpe_idx]
    ax.scatter(
        max_sharpe['Volatility'] * 100,
        max_sharpe['Return'] * 100,
        color=COLORS['green'], s=200, zorder=5,
        marker='*', label='Max Sharpe Portfolio'
    )

    # Individual assets
    for ticker in returns.columns:
        ret = returns[ticker].dropna()
        a_ret = annualised_return(ret) * 100
        a_vol = annualised_volatility(ret) * 100
        color = COLORS.get(ticker, '#FFFFFF')
        label = tickers.get(ticker, ticker)
        ax.scatter(a_vol, a_ret, color=color, s=80,
                   zorder=6, edgecolors='white', linewidths=0.5)
        ax.annotate(label, (a_vol, a_ret),
                    textcoords='offset points', xytext=(6, 3),
                    fontsize=7.5, color=color)

    ax.set_xlabel('Annualised Volatility (%)')
    ax.set_ylabel('Annualised Return (%)')
    ax.set_title('Monte Carlo Efficient Frontier (3,000 Random Portfolios)', pad=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {out_path}")


def plot_drawdowns(prices: pd.DataFrame,
                   tickers: dict,
                   out_path: str):
    """Plot underwater equity curves (drawdown over time)."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for ticker in prices.columns:
        color = COLORS.get(ticker, '#FFFFFF')
        label = tickers.get(ticker, ticker)
        drawdown = (prices[ticker] / prices[ticker].cummax() - 1) * 100
        ax.fill_between(drawdown.index, drawdown, 0,
                         alpha=0.25, color=color)
        ax.plot(drawdown.index, drawdown,
                color=color, linewidth=1.0, label=label)

    ax.set_title('Drawdown from Peak (%)', pad=12)
    ax.set_ylabel('Drawdown (%)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print(f"  Saved: {out_path}")


def print_metrics_table(df: pd.DataFrame):
    """Print a formatted metrics summary to the console."""
    fmt = {
        'Ann. Return':     '{:.1%}',
        'Ann. Volatility': '{:.1%}',
        'Sharpe Ratio':    '{:.2f}',
        'Max Drawdown':    '{:.1%}',
        'VaR (95%)':       '{:.2%}',
        'CVaR (95%)':      '{:.2%}',
    }
    print("\n" + "═" * 72)
    print("  PORTFOLIO METRICS SUMMARY")
    print("═" * 72)
    display = df.copy()
    for col, f in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f.format(x))
    print(display.to_string())
    print("═" * 72 + "\n")


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

def main():
    apply_dark_style()
    out = 'outputs'
    os.makedirs(out, exist_ok=True)

    print("═" * 50)
    print("  FINANCIAL DATA ANALYSIS")
    print("  github.com/benrayner278-lgtm")
    print("═" * 50 + "\n")

    # 1. Load data
    prices = load_data(TICKERS, PERIOD)
    prices = prices[[t for t in TICKERS if t in prices.columns]]
    returns = compute_returns(prices)

    # 2. Metrics
    metrics = build_metrics_table(prices, returns, TICKERS)
    print_metrics_table(metrics)
    metrics.to_csv(f'{out}/metrics_summary.csv')

    # 3. Plots
    print("Generating charts...")
    plot_normalised_prices(prices, TICKERS,
                           f'{out}/01_normalised_prices.png')
    plot_return_distributions(returns, TICKERS,
                              f'{out}/02_return_distributions.png')
    plot_correlation_heatmap(returns, TICKERS,
                             f'{out}/03_correlation_heatmap.png')
    plot_rolling_sharpe(returns, TICKERS,
                        f'{out}/04_rolling_sharpe.png')
    plot_drawdowns(prices, TICKERS,
                   f'{out}/05_drawdowns.png')

    # 4. Efficient frontier
    print("\nRunning Monte Carlo portfolio simulation (3,000 portfolios)...")
    portfolios = simulate_portfolios(returns)
    plot_efficient_frontier(portfolios, returns, TICKERS,
                            f'{out}/06_efficient_frontier.png')

    print("\nAll outputs saved to /outputs/")
    print("Run complete.\n")


if __name__ == '__main__':
    main()