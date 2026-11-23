"""
Experiment 3 - Portfolio allocation over the DJIA constituents with DDPG and TD3
(paper Section 4.5, Fig. 5, Table 2 - second number in each DRL cell).

Instead of buying/selling shares, the agent outputs portfolio *weights* every day
(softmax-normalised by ``StockPortfolioEnv``). The state is the 252-day covariance
matrix of returns plus technical indicators. Evaluation window, initial capital
and baselines are the same as in experiment 2.

Notes / deviations from the stock FinRL code (the library itself is untouched):
* ``StockPortfolioEnv`` returns the raw portfolio value as the reward, which is of
  the order of 1e6 and hard for off-policy agents to learn from. ``ScaledRewardEnv``
  below uses the paper's first reward option instead: the change in portfolio
  value, scaled by ``reward_scaling`` (same convention as ``StockTradingEnv``).
* ``StockPortfolioEnv`` does not charge ``transaction_cost_pct`` when re-weighting,
  so this experiment is effectively frictionless.

Figure 5 and Table 2 combine this experiment with exp2 (multiple stock trading);
they are written to ``outputs/fig5_cumulative_return.png`` and ``outputs/table2.png``
and are complete once both experiments have been run (see ``fig5_table2.py``).

Usage (from the FinRL repo root, with ``pip install -e .`` done):

    python examples/paper_replication/exp3_portfolio_allocation.py
    python examples/paper_replication/exp3_portfolio_allocation.py --timesteps 2000   # smoke test
"""

from __future__ import annotations

import argparse

import pandas as pd

import common
from common import MODEL_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--algos", nargs="+", default=["ddpg", "td3"])
parser.add_argument("--timesteps", type=int, default=100_000, help="training steps per agent")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--initial", type=float, default=1_000_000)
args = parser.parse_args()

common.setup_workdir()

from finrl.agents.stablebaselines3.models import DRLAgent  # noqa: E402
from finrl.config import INDICATORS  # noqa: E402
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv  # noqa: E402


class ScaledRewardEnv(StockPortfolioEnv):
    """``StockPortfolioEnv`` with reward = scaled change in portfolio value."""

    def step(self, actions):
        previous_value = self.portfolio_value
        state, _, terminal, truncated, info = super().step(actions)
        reward = (self.portfolio_value - previous_value) * self.reward_scaling
        return state, reward, terminal, truncated, info


# Data: indicators + rolling covariance matrix, then train/trade split
processed = common.build_dataset(name="djia", tickers=common.DJIA_PAPER_ERA_TICKERS)
processed = common.add_covariance_state(processed)
train, trade = common.split_train_trade(processed)

n = train["tic"].nunique()
env_kwargs = {
    "stock_dim": n,
    "hmax": 100,
    "initial_amount": args.initial,
    "transaction_cost_pct": 0.001,
    "reward_scaling": 1e-4,
    "state_space": n,
    "action_space": n,
    "tech_indicator_list": INDICATORS,
}

results = {}
for algo in args.algos:
    print(f"\n########## training {algo.upper()} (portfolio allocation) ##########")
    env_train, _ = ScaledRewardEnv(df=train, **env_kwargs).get_sb_env()
    model = common.train_agent(algo, env_train, args.timesteps, args.seed)
    model.save(str(MODEL_DIR / f"exp3_{algo}"))

    trade_env = ScaledRewardEnv(df=trade, **env_kwargs)
    returns, weights = DRLAgent.DRL_prediction(model=model, environment=trade_env)
    # The portfolio env only stores daily returns; rebuild the value curve from them.
    values = args.initial * (1 + returns.set_index("date")["daily_return"]).cumprod()
    results[f"{algo.upper()} (Portfolio Allocation)"] = values
    weights.to_csv(common.OUT_DIR / f"exp3_{algo}_weights.csv")

trade_dates = sorted(trade["date"].unique())
baselines = common.static_portfolio_baselines(train, trade, args.initial)
baselines["DJIA"] = common.index_baseline("^DJI", trade_dates, args.initial)

common.report(
    {**results, **baselines},
    name="exp3_portfolio_allocation",
    title="Exp 3 - portfolio allocation on the DJIA (cf. paper Table 2)",
)

# Figure 5 / Table 2 combine this experiment with exp2 (multiple stock trading).
common.build_figure5_table2()
