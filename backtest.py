from typing import Dict, Tuple
import logging

import numpy as np
import pandas as pd

from utils import get_logger, align_to_trading_days


def _prepare_price_frame(price_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = []
    names = []
    for code, df in price_map.items():
        df = align_to_trading_days(df)
        close_col = "close" if "close" in df.columns else "收盘价"
        s = df[close_col].astype(float).rename(code)
        series.append(s)
        names.append(code)
    prices = pd.concat(series, axis=1).sort_index()
    prices = prices.ffill().dropna(how="all")
    return prices


def _normalize_freq(freq: str):
    if freq is None:
        return None
    f = str(freq).strip().upper()
    return f if f in {"M", "Q", "A", "Y", "W", "D"} else None


def _simulate_rebalanced_portfolio(prices: pd.DataFrame, weights: Dict[str, float], freq: str = "M") -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """用持仓模拟月度再平衡组合。
    返回：组合净值、日收益率、各资产持仓价值时间序列。
    """
    # 初始资金
    start_val = 1.0
    norm_freq = _normalize_freq(freq)
    rebal_dates = prices.resample(norm_freq).first().index if norm_freq else pd.DatetimeIndex([])

    # 初始持仓
    holding_value = {c: start_val * weights.get(c, 0.0) for c in prices.columns}

    portfolio_values = []
    per_asset_values = []
    prev_prices = prices.iloc[0]

    for dt, row in prices.iterrows():
        # 再平衡点：按目标权重调整持仓价值（若未启用再平衡，则无此操作）
        if norm_freq and dt in rebal_dates:
            total_val = sum(holding_value.values())
            for c in prices.columns:
                holding_value[c] = total_val * weights.get(c, 0.0)

        # 当日价格变动驱动持仓价值变化
        for c in prices.columns:
            if prev_prices[c] == 0 or np.isnan(prev_prices[c]):
                ret = 0.0
            else:
                ret = (row[c] - prev_prices[c]) / prev_prices[c]
            holding_value[c] *= (1.0 + (ret if not np.isnan(ret) else 0.0))

        total_val = sum(holding_value.values())
        portfolio_values.append((dt, total_val))
        per_asset_values.append((dt, {c: holding_value[c] for c in prices.columns}))
        prev_prices = row

    pf = pd.Series({dt: v for dt, v in portfolio_values}).sort_index()
    asset_df = pd.DataFrame({dt: vals for dt, vals in per_asset_values}).T
    asset_df.index = pf.index
    daily_ret = pf.pct_change().fillna(0.0)
    return pf, daily_ret, asset_df


def backtest(prices_map: Dict[str, pd.DataFrame], weights: Dict[str, float], start_date: str = None, end_date: str = None, freq: str = "M"):
    logger = get_logger("backtest")
    prices = _prepare_price_frame(prices_map)
    if start_date:
        prices = prices.loc[pd.to_datetime(start_date):]
    if end_date:
        prices = prices.loc[:pd.to_datetime(end_date)]
    prices = prices.dropna(how="all")
    logger.info(f"Price frame prepared: {prices.index.min()} -> {prices.index.max()} | {list(prices.columns)}")

    pf, daily_ret, asset_val = _simulate_rebalanced_portfolio(prices, weights, freq=freq)
    return pf, daily_ret, asset_val, prices


def max_drawdown(series: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    cummax = series.cummax()
    drawdown = series / cummax - 1.0
    mdd = drawdown.min()
    end = drawdown.idxmin()
    start = series.loc[:end].idxmax()
    return float(mdd), start, end