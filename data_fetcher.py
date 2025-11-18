from typing import Dict, List
import os
import time
import logging

import pandas as pd
from datetime import timedelta

from utils import (
    get_logger,
    ensure_directories,
    sleep_random_with_log,
    to_ts_code,
    align_to_trading_days,
)
from config import DATA_DIR, REQUEST_INTERVAL_MIN_SECONDS, REQUEST_INTERVAL_MAX_SECONDS, ETF_ADJUST_MODE


def init_tushare(token: str):
    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()
    return ts, pro


def _fetch_slice(ts, pro, ts_code: str, start_date: str, end_date: str, logger: logging.Logger, latest_adj_factor: float | None = None) -> pd.DataFrame:
    """拉取一个小区间的数据。
    使用 fund_adj 的复权因子，与当日收盘价（fund_daily 优先、daily 作为回退）合并，
    在配置为前复权(qfq)时计算：前复权收盘价 = 当日收盘价 ÷ 当日复权因子。
    """
    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")
    logger.info(f"Fetching slice {ts_code} from {start} to {end}...")
    df = None
    try:
        # 1) 复权因子：基金/ETF复权因子（每日）
        logger.info("Using fund_adj for adj_factor")
        adj = pro.fund_adj(ts_code=ts_code, start_date=start, end_date=end)
        if adj is None or adj.empty:
            logger.warning("fund_adj returned empty factors; will use raw close only")

        # 2) 当日收盘价：基金/ETF日线；为空时回退到 daily
        logger.info("Fetching fund_daily close (preferred)")
        px = pro.fund_daily(ts_code=ts_code, start_date=start, end_date=end, fields="ts_code,trade_date,close")
        if px is None or px.empty:
            logger.warning("fund_daily returned empty, falling back to daily for close")
            px = pro.daily(ts_code=ts_code, start_date=start, end_date=end, fields="ts_code,trade_date,close")

        if px is not None and not px.empty:
            px = px.sort_values("trade_date")[['ts_code','trade_date','close']]
            if adj is not None and not adj.empty and 'adj_factor' in adj.columns:
                adj = adj[['ts_code','trade_date','adj_factor']].sort_values('trade_date')
                # 合并并根据配置计算前复权
                df = pd.merge(px, adj, on=['ts_code','trade_date'], how='left')
                # 若有缺失的复权因子，前向填充以提高连续性
                df['adj_factor'] = df['adj_factor'].astype(float)
                df['adj_factor'] = df['adj_factor'].fillna(method='ffill').fillna(method='bfill')
                if str(ETF_ADJUST_MODE).lower() == 'qfq':
                    # 锚定区间最新复权因子，前复权价 = close * adj_factor / latest_adj_factor
                    la = float(latest_adj_factor) if latest_adj_factor and latest_adj_factor > 0 else float(df['adj_factor'].iloc[-1])
                    df['close'] = df['close'].astype(float) * df['adj_factor'] / la
                # 仅保留必要字段
                df = df[["ts_code", "trade_date", "close"]]
            else:
                # 无复权因子，仅返回原始收盘价
                df = px[["ts_code", "trade_date", "close"]]
    except Exception as e:
        logger.exception(f"Tushare request failed for {ts_code} slice {start}-{end}: {e}")
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])  # 返回空
    if df is None or df.empty:
        logger.warning(f"No data returned for {ts_code} slice {start}-{end}")
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])  # 返回空
    return df


def _date_slices(start_date: str, end_date: str, step_days: int = 365) -> List[tuple]:
    """将区间拆分为若干片段，避免接口一次返回受限。"""
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    slices = []
    cur = start_dt
    while cur <= end_dt:
        nxt = min(cur + timedelta(days=step_days - 1), end_dt)
        slices.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + timedelta(days=1)
    return slices


def fetch_daily_close(ts, pro, ts_code: str, start_date: str, end_date: str, logger: logging.Logger) -> pd.DataFrame:
    """分页抓取并合并，确保得到完整区间数据。"""
    frames: List[pd.DataFrame] = []
    # 预先获取整个区间的最新复权因子，用于锚定
    try:
        adj_all = pro.fund_adj(ts_code=ts_code, start_date=pd.to_datetime(start_date).strftime('%Y%m%d'), end_date=pd.to_datetime(end_date).strftime('%Y%m%d'))
        latest_adj_factor = float(adj_all.sort_values('trade_date')['adj_factor'].iloc[-1]) if adj_all is not None and not adj_all.empty else None
    except Exception:
        latest_adj_factor = None
    for s, e in _date_slices(start_date, end_date, step_days=365):
        df = _fetch_slice(ts, pro, ts_code, s, e, logger, latest_adj_factor=latest_adj_factor)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        logger.warning(f"No data returned for full range {ts_code} {start_date}~{end_date}")
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])  # 空占位
    df_all = pd.concat(frames, axis=0)
    df_all = df_all.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")
    df_all = align_to_trading_days(df_all)
    return df_all


def save_to_csv(df: pd.DataFrame, ts_code: str, logger: logging.Logger) -> str:
    ensure_directories()
    out_path = os.path.join(DATA_DIR, f"{ts_code.replace('.', '_')}.csv")
    df_out = df.reset_index()
    df_out.rename(columns={"trade_date": "交易日期", "ts_code": "ETF代码", "close": "收盘价"}, inplace=True)
    # 统一字段，如果缺ts_code则添加
    if "ETF代码" not in df_out.columns:
        df_out["ETF代码"] = ts_code
    if "收盘价" not in df_out.columns and "close" in df_out.columns:
        df_out["收盘价"] = df_out["close"]
    if "交易日期" in df_out.columns:
        df_out["交易日期"] = pd.to_datetime(df_out["交易日期"]).dt.strftime("%Y-%m-%d")
    df_out = df_out[["交易日期", "ETF代码", "收盘价"]]

    # 收盘价统一保留三位小数（数值类型；不转成字符串，避免后续读入时报错）
    df_out["收盘价"] = pd.to_numeric(df_out["收盘价"], errors="coerce").round(3)

    # 若已有文件，进行合并去重，避免覆盖历史数据
    if os.path.exists(out_path):
        try:
            old = pd.read_csv(out_path)
            merged = pd.concat([old, df_out], axis=0)
            merged["交易日期"] = pd.to_datetime(merged["交易日期"]).dt.strftime("%Y-%m-%d")
            merged = merged.drop_duplicates(subset=["交易日期"]).sort_values("交易日期")
            # 合并后同样统一三位小数
            merged["收盘价"] = pd.to_numeric(merged["收盘价"], errors="coerce").round(3)
            df_out = merged
        except Exception as e:
            logger.warning(f"Failed to merge existing CSV for {ts_code}: {e}. Overwriting with new data.")

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {ts_code} CSV -> {out_path}")
    return out_path


def fetch_and_save_many(token: str, codes: List[str], start_date: str, end_date: str) -> Dict[str, str]:
    logger = get_logger("fetch")
    ensure_directories()
    ts, pro = init_tushare(token)
    outputs: Dict[str, str] = {}
    for raw in codes:
        ts_code = to_ts_code(raw)
        df = fetch_daily_close(ts, pro, ts_code, start_date, end_date, logger)
        path = save_to_csv(df, ts_code, logger)
        outputs[ts_code] = path
        # 随机间隔以降低限流风险
        sleep_random_with_log(REQUEST_INTERVAL_MIN_SECONDS, REQUEST_INTERVAL_MAX_SECONDS, logger)
    return outputs