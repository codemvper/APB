from typing import Dict
import os

import numpy as np
import pandas as pd

from utils import get_logger, ensure_directories
from config import REPORT_DIR, RISK_FREE_ANNUAL


def compute_metrics(nav: pd.Series, daily_ret: pd.Series, risk_free_annual: float = RISK_FREE_ANNUAL) -> Dict[str, float]:
    logger = get_logger("report")
    if nav.empty:
        raise ValueError("NAV series is empty")
    start, end = nav.index.min(), nav.index.max()
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    years = (end - start).days / 365.25
    cagr = (1.0 + total_return) ** (1.0 / max(years, 1e-8)) - 1.0

    vol = float(np.std(daily_ret, ddof=1) * np.sqrt(252))
    mean_daily = float(np.mean(daily_ret))
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    sharpe = (mean_daily - rf_daily) / (np.std(daily_ret, ddof=1) + 1e-12) * np.sqrt(252)

    cummax = nav.cummax()
    drawdown = nav / cummax - 1.0
    mdd = float(drawdown.min())

    metrics = {
        "开始日期": str(start.date()),
        "结束日期": str(end.date()),
        "总收益": total_return,
        "年化收益率": cagr,
        "波动率": vol,
        "夏普比率": sharpe,
        "最大回撤": mdd,
        "样本交易日数": int(len(daily_ret)),
    }
    logger.info(
        f"Metrics: AR={metrics['年化收益率']:.2%}, Vol={metrics['波动率']:.2%}, Sharpe={metrics['夏普比率']:.2f}, MDD={metrics['最大回撤']:.2%}"
    )
    return metrics


def save_report(metrics: Dict[str, float], path: str) -> str:
    ensure_directories()
    lines = [
        f"# 回测报告\n",
        f"- 开始日期: {metrics['开始日期']}\n",
        f"- 结束日期: {metrics['结束日期']}\n",
        f"- 总收益: {metrics['总收益']:.2%}\n",
        f"- 年化收益率: {metrics['年化收益率']:.2%}\n",
        f"- 波动率: {metrics['波动率']:.2%}\n",
        f"- 夏普比率: {metrics['夏普比率']:.2f}\n",
        f"- 最大回撤: {metrics['最大回撤']:.2%}\n",
        f"- 样本交易日数: {metrics['样本交易日数']}\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return path


def generate_markdown_report(nav: pd.Series, daily_ret: pd.Series, out_name: str = "backtest_report.md") -> str:
    metrics = compute_metrics(nav, daily_ret)
    out_path = os.path.join(REPORT_DIR, out_name)
    return save_report(metrics, out_path)