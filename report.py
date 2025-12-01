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


def save_holdings_csv(asset_values: pd.DataFrame, capital: float = 1.0, out_name: str = "holdings_daily.csv") -> str:
    ensure_directories()
    df = asset_values.copy()
    df = df.sort_index()
    df = df * float(capital)
    df.insert(0, "date", df.index)
    out_path = os.path.join(REPORT_DIR, out_name)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def save_events_csv(events_df: pd.DataFrame, out_name: str = "rebalance_events.csv") -> str:
    ensure_directories()
    if events_df is None or events_df.empty or (isinstance(events_df, pd.DataFrame) and 'date' not in events_df.columns):
        return ""
    df = events_df.copy()
    df = df.sort_values("date")
    out_path = os.path.join(REPORT_DIR, out_name)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def compute_yearly_metrics(nav: pd.Series, daily_ret: pd.Series, risk_free_annual: float = RISK_FREE_ANNUAL) -> pd.DataFrame:
    if nav is None or nav.empty:
        raise ValueError("NAV series is empty")
    years = sorted(set(pd.to_datetime(nav.index).year))
    rows = []
    rf_daily = (1.0 + float(risk_free_annual)) ** (1.0 / 252.0) - 1.0
    for y in years:
        nav_y = nav[nav.index.year == y]
        ret_y = daily_ret[daily_ret.index.year == y]
        if nav_y.empty or ret_y.empty:
            continue
        start = nav_y.index.min()
        end = nav_y.index.max()
        total_return = float(nav_y.iloc[-1] / nav_y.iloc[0] - 1.0)
        years_len = max(((end - start).days) / 365.25, 1e-8)
        cagr = (1.0 + total_return) ** (1.0 / years_len) - 1.0
        vol = float(np.std(ret_y, ddof=1) * np.sqrt(252))
        mean_daily = float(np.mean(ret_y))
        sharpe = (mean_daily - rf_daily) / (np.std(ret_y, ddof=1) + 1e-12) * np.sqrt(252)
        cummax = nav_y.cummax()
        drawdown = nav_y / cummax - 1.0
        mdd = float(drawdown.min())
        rows.append({
            "年份": int(y),
            "总收益": total_return,
            "年化收益率": cagr,
            "波动率": vol,
            "夏普比率": sharpe,
            "最大回撤": mdd,
            "样本交易日数": int(len(ret_y)),
            "开始日期": str(start.date()),
            "结束日期": str(end.date()),
        })
    return pd.DataFrame(rows).sort_values("年份")


def save_yearly_metrics(df: pd.DataFrame, out_name: str = "yearly_metrics.csv") -> str:
    ensure_directories()
    out_path = os.path.join(REPORT_DIR, out_name)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
