import argparse
from typing import Dict
import os

import pandas as pd

from config import (
    TUSHARE_TOKEN,
    ETF_WEIGHTS,
    START_DATE,
    END_DATE,
    REBALANCE_FREQ,
    DATA_DIR,
    REQUEST_INTERVAL_MIN_SECONDS,
    REQUEST_INTERVAL_MAX_SECONDS,
)
from utils import get_logger, ensure_directories, to_ts_code, sleep_random_with_log
from data_fetcher import init_tushare, fetch_and_save_many, fetch_daily_close, save_to_csv
from validator import check_completeness, detect_anomalies, cross_validate_with_akshare
from backtest import backtest
from report import generate_markdown_report
from visualization import make_portfolio_figure, save_figure_html


def load_csv_close(ts_code: str) -> pd.DataFrame:
    import os
    from config import DATA_DIR
    path = os.path.join(DATA_DIR, f"{ts_code.replace('.', '_')}.csv")
    df = pd.read_csv(path)
    df.rename(columns={"交易日期": "trade_date", "ETF代码": "ts_code", "收盘价": "close"}, inplace=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.sort_values("trade_date", inplace=True)
    return df.set_index("trade_date")


def stage_fetch(args):
    logger = get_logger("stage_fetch")
    codes = [to_ts_code(c) for c in args.codes]
    fetch_and_save_many(TUSHARE_TOKEN, codes, args.start, args.end)
    logger.info("Fetch stage completed.")


def stage_update(args):
    logger = get_logger("stage_update")
    ensure_directories()
    ts, pro = init_tushare(TUSHARE_TOKEN)
    codes = [to_ts_code(c) for c in args.codes]
    # 若未指定结束日期，使用今天
    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    for code in codes:
        start_date = args.start
        path = os.path.join(DATA_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(path):
            try:
                df_old = pd.read_csv(path)
                if not df_old.empty and "交易日期" in df_old.columns:
                    last_dt = pd.to_datetime(df_old["交易日期"]).max().date()
                    # 从下一交易日开始增量
                    start_date = (pd.Timestamp(last_dt) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    logger.info(f"{code} 已有至 {last_dt}, 增量起始 {start_date} -> {end_date}")
            except Exception as e:
                logger.warning(f"读取历史CSV失败 {code}: {e}，改用 --start 作为起点")
        # 若起始超过结束，则无需更新
        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            logger.info(f"{code} 无需更新：已有数据已覆盖到 {end_date}")
            continue
        df_new = fetch_daily_close(ts, pro, code, start_date, end_date, logger)
        if df_new is None or df_new.empty:
            logger.info(f"{code} 无增量数据")
        else:
            save_to_csv(df_new, code, logger)
        sleep_random_with_log(REQUEST_INTERVAL_MIN_SECONDS, REQUEST_INTERVAL_MAX_SECONDS, logger)
    logger.info("Update stage completed.")


def stage_validate(args):
    logger = get_logger("stage_validate")
    import tushare as ts
    ts, pro = init_tushare(TUSHARE_TOKEN)
    codes = [to_ts_code(c) for c in args.codes]
    for code in codes:
        df = load_csv_close(code)
        ok, missing_df = check_completeness(df.reset_index(), pro, args.start, args.end, logger)
        anomalies = detect_anomalies(df.reset_index(), logger)
        cross_ok, merged = cross_validate_with_akshare(code, df.reset_index(), args.start, args.end, logger)
        logger.info(f"Validation {code}: completeness={ok}, anomalies={len(anomalies)}, cross_ok={cross_ok}")
    logger.info("Validation stage completed.")


def stage_backtest(args):
    logger = get_logger("stage_backtest")
    # 准备价格映射
    price_map: Dict[str, pd.DataFrame] = {}
    for code in [to_ts_code(c) for c in args.codes]:
        df = load_csv_close(code)
        price_map[code] = df

    # 权重
    weights = {to_ts_code(k): v for k, v in ETF_WEIGHTS.items()} if args.use_default_weights else {}
    if not weights:
        # 若用户自定义权重，通过 --weights 传入，例如 "511010.SH=0.3,511880.SH=0.25,..."
        for kv in (args.weights or "").split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split("=")
            weights[to_ts_code(k)] = float(v)

    pf_nav, daily_ret, asset_val, prices = backtest(price_map, weights, start_date=args.start, end_date=args.end, freq=args.rebalance)

    # 报告
    report_path = generate_markdown_report(pf_nav, daily_ret)
    logger.info(f"Report saved to {report_path}")

    # 图表
    # 构建分资产净值（以初始1按权重分配）
    asset_navs = {}
    for c in prices.columns:
        s = prices[c].pct_change().fillna(0)
        nav = (1 + s).cumprod()
        nav.iloc[0] = 1.0
        asset_navs[c] = nav
    fig = make_portfolio_figure(pf_nav, asset_navs)
    html_path = save_figure_html(fig)
    logger.info(f"Chart saved to {html_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="全天候组合：数据获取-验证-回测-图表-报告 一体化")
    parser.add_argument("--action", choices=["fetch", "update", "validate", "backtest", "all"], default="all")
    parser.add_argument("--codes", nargs="*", default=["511010", "511880", "510300", "513100", "518880"], help="ETF代码列表，支持无后缀")
    parser.add_argument("--start", default=START_DATE, help="开始日期，YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE, help="结束日期，YYYY-MM-DD")
    parser.add_argument("--rebalance", default=REBALANCE_FREQ, help="再平衡频率：M月、Q季、A年或NONE不再平衡")
    parser.add_argument("--use_default_weights", action="store_true", help="是否使用config中默认权重")
    parser.add_argument("--weights", default="", help="自定义权重，如 511010.SH=0.3,511880.SH=0.25,...")
    return parser


def main():
    ensure_directories()
    logger = get_logger("main")
    parser = build_parser()
    args = parser.parse_args()

    if args.action in ("fetch", "all"):
        stage_fetch(args)
    if args.action == "update":
        stage_update(args)
    if args.action in ("validate", "all"):
        stage_validate(args)
    if args.action in ("backtest", "all"):
        if not args.use_default_weights and not args.weights:
            # 默认使用配置权重
            args.use_default_weights = True
        stage_backtest(args)

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()