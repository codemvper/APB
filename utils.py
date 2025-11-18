import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import time
import random

import pandas as pd

from config import LOG_DIR, DATA_DIR, CHART_DIR, REPORT_DIR


def ensure_directories():
    for d in (LOG_DIR, DATA_DIR, CHART_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)


def get_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(ch_fmt)

    # Rotating file handler
    fh = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(fh_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def to_ts_code(code: str) -> str:
    """为常见沪市ETF代码补后缀；如已含后缀则原样返回。"""
    code = code.strip()
    if code.endswith(".SH") or code.endswith(".SZ"):
        return code
    # 大多数以5开头的ETF在上交所
    if code.startswith("5"):
        return f"{code}.SH"
    # 其它默认深交所
    return f"{code}.SZ"


def sleep_with_log(seconds: float, logger: Optional[logging.Logger] = None):
    if logger:
        logger.debug(f"Rate-limit sleep {seconds:.2f}s...")
    time.sleep(seconds)


def sleep_random_with_log(min_seconds: float, max_seconds: float, logger: Optional[logging.Logger] = None) -> float:
    """在[min_seconds, max_seconds]之间随机等待并记录。返回实际等待秒数。"""
    a = max(0.0, float(min_seconds))
    b = max(a, float(max_seconds))
    delay = random.uniform(a, b)
    if logger:
        logger.debug(f"Rate-limit random sleep {delay:.2f}s (range {a}-{b})...")
    time.sleep(delay)
    return delay


def align_to_trading_days(df: pd.DataFrame, date_col: str = "trade_date") -> pd.DataFrame:
    """确保trade_date为升序日期索引，并去重。"""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.sort_values(date_col).drop_duplicates(subset=[date_col])
        out = out.set_index(date_col)
    else:
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
    return out