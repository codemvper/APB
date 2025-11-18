from typing import Dict, Optional
import os

import pandas as pd
import plotly.graph_objects as go

from utils import get_logger, ensure_directories
from config import CHART_DIR
from backtest import max_drawdown


def _cum_from_returns(ret: pd.Series) -> pd.Series:
    nav = (1.0 + ret).cumprod()
    nav.iloc[0] = 1.0
    return nav


def make_portfolio_figure(portfolio_nav: pd.Series, asset_navs: Optional[Dict[str, pd.Series]] = None, title: str = "组合净值与最大回撤") -> go.Figure:
    logger = get_logger("viz")
    fig = go.Figure()
    # 组合
    fig.add_trace(go.Scatter(x=portfolio_nav.index, y=portfolio_nav.values, name="组合净值", mode="lines", line=dict(width=2)))

    # 子资产
    if asset_navs:
        for name, s in asset_navs.items():
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines", line=dict(width=1, dash="dot")))

    # 最大回撤标注
    mdd, peak_date, trough_date = max_drawdown(portfolio_nav)
    peak_val = float(portfolio_nav.loc[peak_date])
    trough_val = float(portfolio_nav.loc[trough_date])
    fig.add_trace(
        go.Scatter(
            x=[peak_date, trough_date],
            y=[peak_val, trough_val],
            mode="markers+text",
            text=["峰值", f"谷值\nMDD={mdd:.2%}"],
            textposition=["top center", "bottom center"],
            name="回撤标注",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        yaxis_title="净值",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    logger.info(f"Figure prepared. MDD={mdd:.2%} from {peak_date.date()} to {trough_date.date()}")
    return fig


def save_figure_html(fig: go.Figure, out_name: str = "portfolio.html") -> str:
    ensure_directories()
    out_path = os.path.join(CHART_DIR, out_name)
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path