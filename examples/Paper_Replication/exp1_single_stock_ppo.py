"""
Experiment 1 - Single stock trading with PPO (paper Section 4.5, Fig. 4, Table 1).

For each of SPY, QQQ, GOOGL, AMZN, AAPL and MSFT a separate PPO agent is trained
on data before 2019-01-01 and then trades 2019-01-01 .. 2020-09-23 starting from
$100,000. Results are compared against the S&P 500 index (as in the paper) and,
as an extra reference, against simply buying and holding each ticker.

Outputs (in ``examples/paper_replication/outputs/``):
    exp1_fig4_cumulative_return.png   - Figure 4: cumulative return of each agent + S&P 500
    exp1_table1.csv / .png            - Table 1: the six metrics per ticker + S&P 500
    exp1_table1_vs_paper.csv          - our Table 1 next to the numbers printed in the paper

Usage (from the FinRL repo root, with ``pip install -e .`` done):

    python examples/paper_replication/exp1_single_stock_ppo.py              # full run
    python examples/paper_replication/exp1_single_stock_ppo.py --timesteps 2000 --tickers AAPL   # smoke test
"""

from __future__ import annotations

import argparse

import pandas as pd

import common
from common import MODEL_DIR, OUT_DIR

TABLE1_COLUMNS = ["SPY", "QQQ", "GOOGL", "AMZN", "AAPL", "MSFT", "S&P 500"]
FIG4_LEGEND_ORDER = ["AMZN", "GOOGL", "QQQ", "SPY", "AAPL", "MSFT", "S&P 500"]
FIG4_COLORS = {
    "AMZN": "#636EFA",
    "GOOGL": "#EF553B",
    "QQQ": "#00CC96",
    "SPY": "#AB63FA",
    "AAPL": "#FFA15A",
    "MSFT": "#19D3F3",
    "S&P 500": "#B6544F",
}

# Table 1 exactly as printed in the paper (returns, std and drawdown in percent).
PAPER_TABLE1 = pd.DataFrame(
    {
        "SPY": [100_000, 127_044, 14.89, 9.63, 1.49, 20.93],
        "QQQ": [100_000, 163_647, 32.33, 27.51, 1.16, 28.26],
        "GOOGL": [100_000, 174_825, 37.40, 33.41, 1.12, 27.76],
        "AMZN": [100_000, 192_031, 44.94, 29.62, 1.40, 21.13],
        "AAPL": [100_000, 173_063, 36.88, 25.84, 1.35, 22.47],
        "MSFT": [100_000, 172_797, 36.49, 33.41, 1.10, 28.11],
        "S&P 500": [100_000, 133_402, 17.81, 27.00, 0.74, 33.92],
    },
    index=common.METRIC_ROWS,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tickers",
    nargs="+",
    default=["SPY", "QQQ", "GOOGL", "AMZN", "AAPL", "MSFT"],
    help="tickers from Table 1 of the paper",
)
parser.add_argument("--algo", default="ppo", help="paper uses ppo for this experiment")
parser.add_argument("--timesteps", type=int, default=100_000, help="training steps per ticker")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--initial", type=float, default=100_000, help="starting cash (paper: 100,000)")
args = parser.parse_args()

common.setup_workdir()

# Imported after ``setup_workdir`` so any relative output paths land in outputs/.
from finrl.agents.stablebaselines3.models import DRLAgent  # noqa: E402
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # noqa: E402

drl_results = {}
buy_hold_results = {}
trade_dates = None

for tic in args.tickers:
    print(f"\n########## {tic} ##########")
    processed = common.build_dataset(name=f"single_{tic}", tickers=[tic])
    train, trade = common.split_train_trade(processed)
    if trade_dates is None:
        trade_dates = sorted(trade["date"].unique())

    # Train
    env_train, _ = StockTradingEnv(
        df=train, **common.stock_env_kwargs(train, args.initial)
    ).get_sb_env()
    model = common.train_agent(args.algo, env_train, args.timesteps, args.seed)
    model.save(str(MODEL_DIR / f"exp1_{args.algo}_{tic}"))

    # Trade on the held-out window
    trade_env = StockTradingEnv(df=trade, **common.stock_env_kwargs(trade, args.initial))
    account, _ = DRLAgent.DRL_prediction(model=model, environment=trade_env)
    drl_results[tic] = account.set_index("date")["account_value"]
    buy_hold_results[f"Buy&Hold {tic}"] = common.buy_and_hold(trade, tic, args.initial)

sp500 = common.index_baseline("^GSPC", trade_dates, args.initial)

values = pd.concat({**drl_results, "S&P 500": sp500}, axis=1).sort_index().ffill().dropna()
values = values[[c for c in TABLE1_COLUMNS if c in values.columns]]
values.to_csv(OUT_DIR / "exp1_account_values.csv")

# Table 1
table = pd.DataFrame({c: common.perf_stats(values[c]) for c in values.columns})
table.to_csv(OUT_DIR / "exp1_table1.csv")
pretty = common.format_metrics_table(table)
period = f"{common.TRADE_START.replace('-', '/')}-{common.TRADE_END_INCLUSIVE.replace('-', '/')}"
common.save_table_png(
    pretty,
    OUT_DIR / "exp1_table1.png",
    f"Single stock trading using {args.algo.upper()}",
    corner=period,
)

paper = PAPER_TABLE1[[c for c in table.columns if c in PAPER_TABLE1.columns]]
comparison = pd.concat({"ours": table, "paper": paper}, axis=1).swaplevel(axis=1)
comparison = comparison[[(c, s) for c in table.columns for s in ("ours", "paper")]]
comparison.to_csv(OUT_DIR / "exp1_table1_vs_paper.csv")

# Figure 4
common.plot_cumulative_returns(
    values,
    OUT_DIR / "exp1_fig4_cumulative_return.png",
    order=FIG4_LEGEND_ORDER,
    colors=FIG4_COLORS,
    yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    dashed=("S&P 500",),
)

with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(f"\n===== Table 1 - single stock trading with {args.algo.upper()} ({period}) =====")
    print(pretty.to_string())
    print("\n===== Paper Table 1 =====")
    print(common.format_metrics_table(paper).to_string())
print(f"\n[out] saved exp1_fig4_cumulative_return.png, exp1_table1.csv/.png, "
      f"exp1_table1_vs_paper.csv, exp1_account_values.csv in {OUT_DIR}")

# Extra context: is the agent better than just holding the stock?
common.report(
    {**buy_hold_results, "S&P 500": sp500},
    name="exp1_buy_and_hold_reference",
    title="Exp 1 - buy-and-hold reference for the same tickers",
)
