"""
Figure 5 + Table 2 of the paper from the saved results of experiments 2 and 3.

``exp2_multi_stock_ddpg_td3.py`` and ``exp3_portfolio_allocation.py`` already call
this at the end of their run; use this script to regenerate the figure and table
without retraining.

Outputs (in ``examples/paper_replication/outputs/``):
    fig5_cumulative_return.png   - Figure 5: cumulative return of the four agents, DJIA and min-variance
    table2.png / table2.csv      - Table 2 (orange = multiple stock trading, blue = portfolio allocation)
    table2_paper_layout.csv      - Table 2 in the paper's "multiple stock; portfolio allocation" layout
    table2_vs_paper.csv          - our numbers next to the ones printed in the paper

Usage (from the FinRL repo root):

    python examples/paper_replication/fig5_table2.py
"""

from __future__ import annotations

import common

common.setup_workdir()
common.build_figure5_table2()
