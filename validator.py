from typing import Tuple, Optional
import logging

import numpy as np
import pandas as pd

from utils import get_logger, align_to_trading_days
from config import MAX_ABS_DAILY_RETURN, MAD_THRESHOLD


def _trade_calendar(pro, start_ymd: str, end_ymd: str, exchange: str = "SSE") -> pd.DatetimeIndex:
    cal = pro.trade_cal(exchange=exchange, start_date=start_ymd, end_date=end_ymd)
    cal = cal[cal["is_open"] == 1]["cal_date"]
    idx = pd.to_datetime(cal)
    return pd.DatetimeIndex(idx)


def check_completeness(df: pd.DataFrame, pro, start_date: str, end_date: str, logger: Optional[logging.Logger] = None) -> Tuple[bool, pd.DataFrame]:
    logger = logger or get_logger("validate")
    # df 已是按交易日索引
    df = align_to_trading_days(df)
    start_ymd = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_ymd = pd.to_datetime(end_date).strftime("%Y%m%d")
    cal_idx = _trade_calendar(pro, start_ymd, end_ymd)
    missing = cal_idx.difference(df.index)
    ok = len(missing) == 0
    if not ok:
        logger.warning(f"Missing {len(missing)} open days. Example: {list(missing[:5])}")
    else:
        logger.info("Data completeness: OK (no missing open days)")
    return ok, pd.DataFrame({"missing_date": missing})


def detect_anomalies(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    logger = logger or get_logger("validate")
    df = align_to_trading_days(df)
    close = df["close" if "close" in df.columns else "收盘价"].astype(float)
    ret = close.pct_change()

    # 基于绝对阈值
    flag1 = ret.abs() > MAX_ABS_DAILY_RETURN

    # 基于MAD鲁棒检测
    med = np.nanmedian(ret)
    mad = np.nanmedian(np.abs(ret - med))
    robust_z = np.abs((ret - med) / (mad + 1e-8))
    flag2 = robust_z > MAD_THRESHOLD

    flagged = df.loc[flag1 | flag2].copy()
    flagged["daily_return"] = ret[flag1 | flag2]
    flagged["robust_z"] = robust_z[flag1 | flag2]
    if not flagged.empty:
        logger.warning(f"Detected {len(flagged)} potential anomalies")
    else:
        logger.info("Price reasonableness: OK (no anomalies by rules)")
    return flagged


def cross_validate_with_akshare(ts_code: str, df_ts: pd.DataFrame, start_date: str, end_date: str, logger: Optional[logging.Logger] = None) -> Tuple[bool, Optional[pd.DataFrame]]:
    """使用AkShare的东财ETF数据进行交叉验证。
    可能因环境未安装或网络错误而失败，失败时返回(False, None)。
    """
    logger = logger or get_logger("validate")
    try:
        import akshare as ak
    except Exception as e:
        logger.warning(f"AkShare not available for cross validation: {e}")
        return False, None

    code = ts_code.split(".")[0]
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    try:
        ak_df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
    except Exception as e:
        logger.warning(f"AkShare request failed for {ts_code}: {e}")
        return False, None

    if ak_df is None or ak_df.empty:
        logger.warning(f"AkShare returned empty for {ts_code}")
        return False, None

    ak_df = ak_df[["日期", "收盘"]].rename(columns={"日期": "trade_date", "收盘": "close"})
    ak_df = align_to_trading_days(ak_df)

    # 对齐合并
    ts_df = df_ts[["close"]].copy()
    ts_df = align_to_trading_days(ts_df)
    merged = ts_df.join(ak_df, how="inner", lsuffix="_ts", rsuffix="_ak")
    if merged.empty:
        logger.warning("No overlapping dates for cross validation")
        return False, None

    # 比较误差与相关性
    diff_pct = (merged["close_ts"] - merged["close_ak"]) / merged["close_ak"]
    corr = merged["close_ts"].pct_change().corr(merged["close_ak"].pct_change())
    mean_abs_diff = diff_pct.abs().mean()
    logger.info(f"Cross check {ts_code}: mean_abs_diff={mean_abs_diff:.4%}, corr={corr:.3f}")

    ok = (mean_abs_diff < 0.01) and (corr is not None and corr > 0.95)
    return ok, merged.assign(diff_pct=diff_pct)