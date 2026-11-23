"""
Shared helpers for replicating the experiments in the FinRL paper
(Liu et al., "FinRL: A Deep Reinforcement Learning Library for Automated Stock
Trading in Quantitative Finance", NeurIPS 2020 DRL Workshop, arXiv:2011.09607).

Paper setup that is fixed here
------------------------------
* Trading (test) window : 2019-01-01 .. 2020-09-23  (Section 4.5, Tables 1 & 2)
* Data source           : Yahoo Finance via FinRL's ``YahooDownloader``
* Metrics (Section 4.1) : final portfolio value, annualized return,
                          annualized standard deviation, Sharpe ratio and
                          maximum drawdown

Things the paper does NOT specify (so they are assumptions you can change)
--------------------------------------------------------------------------
* Training start date  -> ``TRAIN_START`` below (2009-01-01, the start date
  used in the public FinRL tutorials).
* Training length / hyper-parameters -> ``--timesteps`` and ``finrl/config.py``.
* No separate validation window is used; training ends where trading begins.

All outputs (cached downloads, trained models, CSVs and plots) are written to
``examples/paper_replication/outputs/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# --------------------------------------------------------------------------- #
# Paths and dates
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
CACHE_DIR = OUT_DIR / "cache"
MODEL_DIR = OUT_DIR / "trained_models"

TRAIN_START = "2009-01-01"  # assumption, see module docstring
TRAIN_END = "2019-01-01"  # exclusive (``data_split`` uses date < end)
TRADE_START = "2019-01-01"
TRADE_END_INCLUSIVE = "2020-09-23"  # last day reported in the paper
TRADE_END = "2020-09-24"  # exclusive upper bound for download + data_split

TRADING_DAYS = 252

# DJIA constituents used by FinRL right after the Aug-2020 index change.
# Yahoo Finance no longer serves some of them (e.g. WBA was taken private) and
# DOW only started trading in 2019-04, so ``FeatureEngineer.clean_data`` will
# automatically drop any ticker that does not have the full date range.
DJIA_PAPER_ERA_TICKERS = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    "DOW",
]


def setup_workdir() -> None:
    """Create the output folders and ``chdir`` into them.

    Some FinRL environments write plots to a relative ``results/`` folder, so
    running from inside ``outputs/`` keeps the repo clean.
    """
    for d in (OUT_DIR, CACHE_DIR, MODEL_DIR, OUT_DIR / "results"):
        d.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT_DIR)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_cached(name: str, tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV with ``YahooDownloader`` and cache it as a CSV."""
    path = CACHE_DIR / f"{name}_{start}_{end}.csv"
    if path.exists():
        print(f"[data] loading cached download {path.name}")
        return pd.read_csv(path)
    print(f"[data] downloading {len(tickers)} ticker(s) from Yahoo Finance ...")
    df = YahooDownloader(start_date=start, end_date=end, ticker_list=tickers).fetch_data()
    df.to_csv(path, index=False)
    return df


def build_dataset(
    name: str, tickers: list[str], use_turbulence: bool = False
) -> pd.DataFrame:
    """Download + add technical indicators (+ optional turbulence index).

    Tickers without data for the whole period are dropped by FinRL's
    ``FeatureEngineer.clean_data``; the surviving universe is printed.
    """
    raw = fetch_cached(name, tickers, TRAIN_START, TRADE_END)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=use_turbulence,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(raw)
    processed = processed.sort_values(["date", "tic"]).reset_index(drop=True)

    kept = sorted(processed["tic"].unique())
    dropped = sorted(set(tickers) - set(kept))
    print(f"[data] universe: {len(kept)} tickers -> {kept}")
    if dropped:
        print(f"[data] dropped (no full-period data on Yahoo): {dropped}")
    return processed


def split_train_trade(processed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training set (< 2019-01-01) and trading set (2019-01-01..2020-09-23)."""
    train = data_split(processed, TRAIN_START, TRAIN_END)
    trade = data_split(processed, TRADE_START, TRADE_END)
    print(
        f"[data] train rows={len(train)} "
        f"({train.date.min()}..{train.date.max()}), "
        f"trade rows={len(trade)} ({trade.date.min()}..{trade.date.max()})"
    )
    return train, trade


def add_covariance_state(df: pd.DataFrame, lookback: int = TRADING_DAYS) -> pd.DataFrame:
    """Add the ``cov_list`` column required by ``StockPortfolioEnv``.

    For every date the covariance matrix of the previous ``lookback`` daily
    returns is attached (same definition as the FinRL portfolio tutorial, but
    computed from a single pivot instead of re-pivoting inside the loop).
    The first ``lookback`` days have no covariance and are removed.
    """
    prices = df.pivot(index="date", columns="tic", values="close")
    returns = prices.pct_change()
    dates = prices.index
    cov_dates, cov_list = [], []
    for i in range(lookback, len(dates)):
        window = returns.iloc[i - lookback + 1 : i + 1]
        cov_dates.append(dates[i])
        cov_list.append(window.cov().values)

    # Same construction as the FinRL portfolio tutorial: an inner merge on date
    # drops the first ``lookback`` days that have no covariance estimate.
    df_cov = pd.DataFrame({"date": cov_dates, "cov_list": cov_list})
    df = df.merge(df_cov, on="date")
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Environments / agents
# --------------------------------------------------------------------------- #
def stock_env_kwargs(
    df: pd.DataFrame, initial_amount: float, hmax: int = 100, cost: float = 0.001
) -> dict:
    """Keyword arguments for ``StockTradingEnv`` (per-share-percentage cost)."""
    n = df["tic"].nunique()
    return {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * n,
        "buy_cost_pct": [cost] * n,
        "sell_cost_pct": [cost] * n,
        "state_space": 1 + 2 * n + len(INDICATORS) * n,
        "stock_dim": n,
        "tech_indicator_list": INDICATORS,
        "action_space": n,
        "reward_scaling": 1e-4,
    }


def train_agent(algo: str, env_train, timesteps: int, seed: int):
    """Train one Stable-Baselines3 agent with the defaults in ``finrl/config.py``."""
    from finrl.agents.stablebaselines3.models import DRLAgent

    agent = DRLAgent(env=env_train)
    model = agent.get_model(algo, seed=seed, verbose=0)
    return agent.train_model(model=model, tb_log_name=algo, total_timesteps=timesteps)


# --------------------------------------------------------------------------- #
# Baselines (Section 4.2)
# --------------------------------------------------------------------------- #
def index_baseline(
    ticker: str, dates: list[str], initial_amount: float
) -> pd.Series:
    """Buy the index (e.g. ``^GSPC``, ``^DJI``) on the first trading day and hold."""
    df = YahooDownloader(
        start_date=TRADE_START, end_date=TRADE_END, ticker_list=[ticker]
    ).fetch_data()
    close = df.set_index("date")["close"].reindex(dates).ffill().bfill()
    return close / close.iloc[0] * initial_amount


def buy_and_hold(trade: pd.DataFrame, tic: str, initial_amount: float) -> pd.Series:
    """Buy a single stock on the first trading day and hold."""
    close = trade[trade.tic == tic].set_index("date")["close"]
    return close / close.iloc[0] * initial_amount


def _long_only_weights(mu: np.ndarray, cov: np.ndarray, objective: str) -> np.ndarray:
    n = len(mu)
    if objective == "min_var":
        fun = lambda w: w @ cov @ w  # noqa: E731
    else:  # "max_sharpe"
        fun = lambda w: -(w @ mu) / np.sqrt(w @ cov @ w)  # noqa: E731
    res = minimize(
        fun,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
    )
    return res.x


def static_portfolio_baselines(
    train: pd.DataFrame, trade: pd.DataFrame, initial_amount: float
) -> dict[str, pd.Series]:
    """Equal-weighted, min-variance and mean-variance (max-Sharpe) baselines.

    Weights are estimated on the *training* returns (long-only, fully invested),
    the portfolio is bought at the first trading-day close and then held
    (no rebalancing), which mirrors the repo's own backtest script.
    """
    train_px = train.pivot(index="date", columns="tic", values="close")
    trade_px = trade.pivot(index="date", columns="tic", values="close")
    rets = train_px.pct_change().dropna() * 100  # percent units keeps SLSQP well scaled
    mu, cov = rets.mean().values, rets.cov().values

    n = trade_px.shape[1]
    weights = {
        "Equal-weight": np.ones(n) / n,
        "Min-variance": _long_only_weights(mu, cov, "min_var"),
        "Mean-variance": _long_only_weights(mu, cov, "max_sharpe"),
    }
    out = {}
    first_px = trade_px.iloc[0].values
    for name, w in weights.items():
        shares = initial_amount * w / first_px
        out[name] = pd.Series(trade_px.values @ shares, index=trade_px.index)
    return out


# --------------------------------------------------------------------------- #
# Metrics + reporting (Section 4.1)
# --------------------------------------------------------------------------- #
def perf_stats(values: pd.Series) -> dict[str, float]:
    """The five metrics from Section 4.1 of the paper, from a daily value series."""
    values = values.astype(float).dropna()
    rets = values.pct_change().dropna()
    n = len(rets)
    ann_ret = (values.iloc[-1] / values.iloc[0]) ** (TRADING_DAYS / n) - 1
    ann_std = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (
        rets.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else np.nan
    )
    max_dd = (values / values.cummax() - 1).min()
    return {
        "Initial value": values.iloc[0],
        "Final value": values.iloc[-1],
        "Annualized return": ann_ret * 100,
        "Annualized Std": ann_std * 100,
        "Sharpe ratio": sharpe,
        "Max drawdown": abs(max_dd) * 100,
    }


METRIC_ROWS = [
    "Initial value",
    "Final value",
    "Annualized return",
    "Annualized Std",
    "Sharpe ratio",
    "Max drawdown",
]


def format_metric(row: str, value: float) -> str:
    """Format one metric the way the paper's tables print it (100,000 / 14.89% / 1.49)."""
    if row in ("Initial value", "Final value"):
        return f"{value:,.0f}"
    if row == "Sharpe ratio":
        return f"{value:.2f}"
    return f"{value:.2f}%"


def format_metrics_table(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {c: [format_metric(r, float(table.loc[r, c])) for r in table.index] for c in table.columns},
        index=table.index,
    )


def save_table_png(
    table: pd.DataFrame,
    path,
    title: str,
    corner: str = "",
    col_colors: dict[str, str] | None = None,
) -> None:
    """Render an already formatted table as a PNG (optionally colouring whole columns)."""
    fig, ax = plt.subplots(figsize=(1.7 * (len(table.columns) + 1.5), 2.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=table.values,
        rowLabels=table.index,
        colLabels=table.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    if corner:
        tbl.add_cell(0, -1, width=tbl[0, 0].get_width(), height=tbl[0, 0].get_height(), text=corner)
    for j, col in enumerate(table.columns):
        color = (col_colors or {}).get(col)
        if color:
            for i in range(len(table.index) + 1):
                tbl[i, j].get_text().set_color(color)
    ax.set_title(title, fontsize=11)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_returns(
    values: pd.DataFrame,
    path,
    order: list[str],
    colors: dict[str, str],
    yticks: list[float],
    dashed: tuple[str, ...] = (),
) -> None:
    """Cumulative return (value / initial - 1) styled after Figures 4 and 5 of the paper."""
    cum = values / values.iloc[0] - 1
    cum.index = pd.to_datetime(cum.index)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name in [n for n in order if n in cum.columns]:
        ax.plot(
            cum.index,
            cum[name],
            label=name,
            color=colors.get(name),
            linewidth=1.4,
            linestyle="--" if name in dashed else "-",
        )

    # Paper ticks are every 10 weeks starting on the first trading day.
    ticks = pd.date_range(cum.index[0], cum.index[-1] + pd.Timedelta(days=1), freq="70D")
    labels = [
        t.strftime("%b %-d\n%Y") if i == 0 or t.year != ticks[i - 1].year else t.strftime("%b %-d")
        for i, t in enumerate(ticks)
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(cum.index[0], cum.index[-1])
    step = yticks[1] - yticks[0]
    ax.set_yticks(yticks)
    ax.set_ylim(
        min(yticks[0], cum.min().min()) - step / 2,
        max(yticks[-1], cum.max().max()) + step / 2,
    )
    ax.set_ylabel("Cumulative Return")
    ax.grid(color="#d9dde8", linewidth=0.8)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 5 / Table 2 (combines experiments 2 and 3)
# --------------------------------------------------------------------------- #
FIG5_ORDER = [
    "TD3 (Portfolio Allocation)",
    "DDPG (Portfolio Allocation)",
    "TD3 (Multiple Stock)",
    "DDPG (Multiple Stock)",
    "DJIA",
    "Min-Variance",
]
FIG5_COLORS = {
    "TD3 (Portfolio Allocation)": "#636EFA",
    "DDPG (Portfolio Allocation)": "#EF553B",
    "TD3 (Multiple Stock)": "#00CC96",
    "DDPG (Multiple Stock)": "#AB63FA",
    "DJIA": "#FFA15A",
    "Min-Variance": "#19D3F3",
}
MULTI_STOCK_COLOR = "#E8730C"  # orange text in Table 2
ALLOCATION_COLOR = "#1A9BD7"  # blue text in Table 2

# Table 2 exactly as printed in the paper (returns, std and drawdown in percent).
PAPER_TABLE2 = pd.DataFrame(
    {
        "TD3 (Multiple Stock)": [1_000_000, 1_403_337, 21.40, 14.60, 1.38, 11.52],
        "TD3 (Portfolio Allocation)": [1_000_000, 1_381_120, 17.61, 17.01, 1.03, 12.78],
        "DDPG (Multiple Stock)": [1_000_000, 1_396_607, 20.34, 15.89, 1.28, 13.72],
        "DDPG (Portfolio Allocation)": [1_000_000, 1_281_120, 15.81, 16.60, 0.98, 13.68],
        "Min-Variance": [1_000_000, 1_171_120, 8.38, 26.21, 0.44, 34.34],
        "DJIA": [1_000_000, 1_185_260, 10.61, 28.63, 0.48, 37.01],
    },
    index=METRIC_ROWS,
)

EXP2_VALUES = OUT_DIR / "exp2_multi_stock_account_values.csv"
EXP3_VALUES = OUT_DIR / "exp3_portfolio_allocation_account_values.csv"


def build_figure5_table2() -> pd.DataFrame | None:
    """Merge the saved account values of experiments 2 and 3 into Figure 5 + Table 2.

    Works with whichever experiments have been run so far; the figure and table
    are complete once both have finished.
    """
    parts = [pd.read_csv(p, index_col=0) for p in (EXP2_VALUES, EXP3_VALUES) if p.exists()]
    if not parts:
        print("[fig5] no results from experiment 2 or 3 yet")
        return None
    values = pd.concat(parts, axis=1).rename(columns={"Min-variance": "Min-Variance"})
    values = values.loc[:, ~values.columns.duplicated()]
    values = values[[c for c in FIG5_ORDER if c in values.columns]].sort_index().ffill().dropna()
    values.to_csv(OUT_DIR / "fig5_account_values.csv")

    table = pd.DataFrame({c: perf_stats(values[c]) for c in values.columns})
    table.to_csv(OUT_DIR / "table2.csv")
    pretty = format_metrics_table(table)
    period = f"{TRADE_START.replace('-', '/')}-{TRADE_END_INCLUSIVE.replace('-', '/')}"

    # Paper layout: one column per algorithm, "multiple stock; portfolio allocation".
    paper_layout = {}
    for algo in ("TD3", "DDPG"):
        cells = [pretty[f"{algo} ({k})"] for k in ("Multiple Stock", "Portfolio Allocation")
                 if f"{algo} ({k})" in pretty.columns]
        if cells:
            paper_layout[algo] = cells[0] if len(cells) == 1 else cells[0] + "; " + cells[1]
    if "Min-Variance" in pretty.columns:
        paper_layout["Min-Var."] = pretty["Min-Variance"]
    if "DJIA" in pretty.columns:
        paper_layout["DJIA"] = pretty["DJIA"]
    paper_layout = pd.DataFrame(paper_layout)
    paper_layout.index.name = period
    paper_layout.to_csv(OUT_DIR / "table2_paper_layout.csv")

    wide_order = [
        "TD3 (Multiple Stock)",
        "TD3 (Portfolio Allocation)",
        "DDPG (Multiple Stock)",
        "DDPG (Portfolio Allocation)",
        "Min-Variance",
        "DJIA",
    ]
    wide = pretty[[c for c in wide_order if c in pretty.columns]]
    wide.columns = [c.replace(" (", "\n(") for c in wide.columns]
    col_colors = {
        c: MULTI_STOCK_COLOR if "Multiple Stock" in c else ALLOCATION_COLOR
        for c in wide.columns
        if "(" in c
    }
    save_table_png(
        wide,
        OUT_DIR / "table2.png",
        "Multiple stock trading (orange) and portfolio allocation (blue) over the DJIA constituents",
        corner=period,
        col_colors=col_colors,
    )

    paper = PAPER_TABLE2[[c for c in table.columns if c in PAPER_TABLE2.columns]]
    comparison = pd.concat({"ours": table, "paper": paper}, axis=1).swaplevel(axis=1)
    comparison = comparison[[(c, s) for c in table.columns for s in ("ours", "paper")]]
    comparison.to_csv(OUT_DIR / "table2_vs_paper.csv")

    plot_cumulative_returns(
        values,
        OUT_DIR / "fig5_cumulative_return.png",
        order=FIG5_ORDER,
        colors=FIG5_COLORS,
        yticks=[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4],
    )

    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(f"\n===== Table 2 ({period}) - multiple stock; portfolio allocation =====")
        print(paper_layout.to_string())
        print("\n===== Paper Table 2 =====")
        print(format_metrics_table(paper).to_string())
    missing = [c for c in FIG5_ORDER if c not in values.columns]
    if missing:
        print(f"[fig5] still missing {missing} - run the other experiment to complete Figure 5")
    print(f"[out] saved fig5_cumulative_return.png, table2.png, table2.csv, "
          f"table2_paper_layout.csv, table2_vs_paper.csv in {OUT_DIR}")
    return table


def report(series: dict[str, pd.Series], name: str, title: str) -> pd.DataFrame:
    """Print a Table-1/2 style comparison, save CSV + PNG, return the table."""
    values = pd.concat(series, axis=1).sort_index().ffill().dropna()
    values.to_csv(OUT_DIR / f"{name}_account_values.csv")

    table = pd.DataFrame({k: perf_stats(values[k]) for k in values.columns})
    table.to_csv(OUT_DIR / f"{name}_metrics.csv")

    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200):
        print(f"\n===== {title} =====")
        print(table.to_string())

    plot_df = values.copy()
    plot_df.index = pd.to_datetime(plot_df.index)
    ax = plot_df.plot(figsize=(12, 5))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(alpha=0.3)
    plt.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[out] saved {name}_metrics.csv, {name}_account_values.csv, {name}.png in {OUT_DIR}")
    return table
