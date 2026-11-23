"""
Experiment 2 - Multiple stock trading on the DJIA with DDPG and TD3
(paper Section 4.5, Fig. 5, Table 2 - first number in each DRL cell).

Agents are trained on data before 2019-01-01 and trade the DJIA constituents from
2019-01-01 to 2020-09-23 starting from $1,000,000, with a 0.1% per-share transaction
cost. They are compared against the DJIA index and the paper's min-variance baseline
(plus equal-weight and mean-variance baselines from Section 4.2).

The financial turbulence index (paper Eq. 1) is used as a risk-aversion switch: when
it exceeds a threshold the environment stops buying and liquidates positions. The
threshold is the ``--turb-quantile`` quantile of the *training* turbulence values
(0.90 is what the paper's ensemble-strategy predecessor uses). Pass ``--no-turbulence``
to disable it.

Figure 5 and Table 2 combine this experiment with exp3 (portfolio allocation);
they are written to ``outputs/fig5_cumulative_return.png`` and ``outputs/table2.png``
and are complete once both experiments have been run (see ``fig5_table2.py``).

Usage (from the FinRL repo root, with ``pip install -e .`` done):

    python examples/paper_replication/exp2_multi_stock_ddpg_td3.py
    python examples/paper_replication/exp2_multi_stock_ddpg_td3.py --timesteps 2000   # smoke test
    python examples/paper_replication/exp2_multi_stock_ddpg_td3.py --algos a2c ppo sac ddpg td3
"""

from __future__ import annotations

import argparse

import numpy as np

import common
from common import MODEL_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--algos", nargs="+", default=["ddpg", "td3"])
parser.add_argument("--timesteps", type=int, default=100_000, help="training steps per agent")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--initial", type=float, default=1_000_000)
parser.add_argument("--no-turbulence", action="store_true", help="disable the turbulence risk switch")
parser.add_argument("--turb-quantile", type=float, default=0.90)
args = parser.parse_args()

common.setup_workdir()

from finrl.agents.stablebaselines3.models import DRLAgent  # noqa: E402
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # noqa: E402

use_turb = not args.no_turbulence
processed = common.build_dataset(
    name="djia", tickers=common.DJIA_PAPER_ERA_TICKERS, use_turbulence=use_turb
)
train, trade = common.split_train_trade(processed)
env_kwargs = common.stock_env_kwargs(train, args.initial)

turbulence_threshold = None
if use_turb:
    train_turb = train[["date", "turbulence"]].drop_duplicates("date")["turbulence"]
    turbulence_threshold = float(np.quantile(train_turb.values, args.turb_quantile))
    print(
        f"[turbulence] threshold = {turbulence_threshold:.2f} "
        f"({args.turb_quantile:.0%} quantile of training turbulence)"
    )

# Train once per algorithm, then trade on the held-out window.
results = {}
for algo in args.algos:
    print(f"\n########## training {algo.upper()} ##########")
    env_train, _ = StockTradingEnv(df=train, **env_kwargs).get_sb_env()
    model = common.train_agent(algo, env_train, args.timesteps, args.seed)
    model.save(str(MODEL_DIR / f"exp2_{algo}"))

    trade_env = StockTradingEnv(
        df=trade, turbulence_threshold=turbulence_threshold, **env_kwargs
    )
    account, _ = DRLAgent.DRL_prediction(model=model, environment=trade_env)
    results[f"{algo.upper()} (Multiple Stock)"] = account.set_index("date")["account_value"]

trade_dates = sorted(trade["date"].unique())
baselines = common.static_portfolio_baselines(train, trade, args.initial)
baselines["DJIA"] = common.index_baseline("^DJI", trade_dates, args.initial)

common.report(
    {**results, **baselines},
    name="exp2_multi_stock",
    title="Exp 2 - multiple stock trading on the DJIA (cf. paper Table 2)",
)

# Figure 5 / Table 2 combine this experiment with exp3 (portfolio allocation).
common.build_figure5_table2()
