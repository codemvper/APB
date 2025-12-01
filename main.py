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
    STRATEGY_MODE,
    DATA_DIR,
    REQUEST_INTERVAL_MIN_SECONDS,
    REQUEST_INTERVAL_MAX_SECONDS,
)
from utils import get_logger, ensure_directories, to_ts_code, sleep_random_with_log
from data_fetcher import init_tushare, fetch_and_save_many, fetch_daily_close, save_to_csv
from validator import check_completeness, detect_anomalies, cross_validate_with_akshare
from backtest import backtest
from report import generate_markdown_report, save_holdings_csv, save_events_csv, compute_metrics
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

    sma_short = int(args.sma50) if hasattr(args, "sma50") and args.sma50 is not None else 30
    sma_mid = int(args.sma100) if hasattr(args, "sma100") and args.sma100 is not None else 80
    sma_long = int(args.sma200) if hasattr(args, "sma200") and args.sma200 is not None else 200
    momentum_window = int(args.momentum_window) if hasattr(args, "momentum_window") and args.momentum_window is not None else 10
    
    pf_nav, daily_ret, asset_val, prices, events = backtest(
        price_map,
        weights,
        start_date=args.start,
        end_date=args.end,
        freq=args.rebalance,
        strategy=args.strategy,
        sma_short=sma_short,
        sma_mid=sma_mid,
        sma_long=sma_long,
        momentum_window=momentum_window,
    )

    # 报告
    report_path = generate_markdown_report(pf_nav, daily_ret)
    logger.info(f"Report saved to {report_path}")
    from report import compute_yearly_metrics, save_yearly_metrics
    yearly_df = compute_yearly_metrics(pf_nav, daily_ret)
    yearly_path = save_yearly_metrics(yearly_df)
    logger.info(f"Yearly metrics saved to {yearly_path}")

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
    holdings_path = save_holdings_csv(asset_val, capital=float(args.capital))
    logger.info(f"Holdings saved to {holdings_path}")
    events_path = save_events_csv(events)
    logger.info(f"Events saved to {events_path}")


def stage_gridsearch(args):
    logger = get_logger("stage_gridsearch")
    price_map: Dict[str, pd.DataFrame] = {}
    for code in [to_ts_code(c) for c in args.codes]:
        df = load_csv_close(code)
        price_map[code] = df

    weights = {to_ts_code(k): v for k, v in ETF_WEIGHTS.items()} if args.use_default_weights else {}
    if not weights:
        for kv in (args.weights or "").split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split("=")
            weights[to_ts_code(k)] = float(v)

    def parse_list(s: str):
        return [int(x) for x in str(s).split(",") if str(x).strip()]

    import os
    from config import REPORT_DIR
    rows = []

    if args.strategy == "momentum":
        # 动量策略网格搜索
        mom_list = parse_list(getattr(args, "momentum_list", "3,6,9,10,12"))
        for mw in mom_list:
            pf_nav, daily_ret, asset_val, prices, events = backtest(
                price_map,
                weights,
                start_date=args.start,
                end_date=args.end,
                freq=args.rebalance,
                strategy="momentum",
                momentum_window=mw
            )
            metrics = compute_metrics(pf_nav, daily_ret)
            rows.append({
                "MomentumWindow": mw,
                "总收益": metrics["总收益"],
                "年化收益率": metrics["年化收益率"],
                "波动率": metrics["波动率"],
                "夏普比率": metrics["夏普比率"],
                "最大回撤": metrics["最大回撤"],
            })
            logger.info(
                f"GS MomWindow={mw} AR={metrics['年化收益率']:.2%} Sharpe={metrics['夏普比率']:.2f} MDD={metrics['最大回撤']:.2%}"
            )
        out_path = os.path.join(REPORT_DIR, "gridsearch_momentum.csv")
    else:
        # T-Value 策略网格搜索
        l50 = parse_list(getattr(args, "sma50_list", "30,40,50,60"))
        l100 = parse_list(getattr(args, "sma100_list", "80,100,120"))
        l200 = parse_list(getattr(args, "sma200_list", "180,200,250"))
        
        for w50 in l50:
            for w100 in l100:
                for w200 in l200:
                    pf_nav, daily_ret, asset_val, prices, events = backtest(
                        price_map,
                        weights,
                        start_date=args.start,
                        end_date=args.end,
                        freq=args.rebalance,
                        strategy="tvalue",
                        sma_short=w50,
                        sma_mid=w100,
                        sma_long=w200,
                    )
                    metrics = compute_metrics(pf_nav, daily_ret)
                    rows.append({
                        "SMA50": w50,
                        "SMA100": w100,
                        "SMA200": w200,
                        "总收益": metrics["总收益"],
                        "年化收益率": metrics["年化收益率"],
                        "波动率": metrics["波动率"],
                        "夏普比率": metrics["夏普比率"],
                        "最大回撤": metrics["最大回撤"],
                    })
                    logger.info(
                        f"GS {w50}-{w100}-{w200} AR={metrics['年化收益率']:.2%} Sharpe={metrics['夏普比率']:.2f} MDD={metrics['最大回撤']:.2%}"
                    )
        out_path = os.path.join(REPORT_DIR, "gridsearch_sma.csv")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"GridSearch saved to {out_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="全天候组合：数据获取-验证-回测-图表-报告 一体化")
    parser.add_argument("--action", choices=["fetch", "update", "validate", "backtest", "gridsearch", "all"], default="all")
    parser.add_argument("--codes", nargs="*", default=["511010", "511880", "510300", "513100", "518880"], help="ETF代码列表，支持无后缀")
    parser.add_argument("--start", default=START_DATE, help="开始日期，YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE, help="结束日期，YYYY-MM-DD")
    parser.add_argument("--rebalance", default=REBALANCE_FREQ, help="再平衡频率：M月、Q季、A年或NONE不再平衡")
    parser.add_argument("--strategy", choices=["fixed", "tvalue", "momentum"], default=STRATEGY_MODE, help="回测策略模式")
    parser.add_argument("--capital", default=1000000, help="初始资金规模，用于持仓明细导出")
    parser.add_argument("--use_default_weights", action="store_true", help="是否使用config中默认权重")
    parser.add_argument("--weights", default="", help="自定义权重，如 511010.SH=0.3,511880.SH=0.25,...")
    parser.add_argument("--sma50", type=int, default=30)
    parser.add_argument("--sma100", type=int, default=80)
    parser.add_argument("--sma200", type=int, default=200)
    parser.add_argument("--sma50_list", default="20,30,40,50,60")
    parser.add_argument("--sma100_list", default="80,90,100,110,120")
    parser.add_argument("--sma200_list", default="180,190,200,210,220,230,240,250")
    parser.add_argument("--momentum_window", type=int, help="绝对动量回顾窗口(月)")
    parser.add_argument("--momentum_list", default="3,6,9,10,12", help="绝对动量网格搜索窗口列表(逗号分隔)")
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
    if args.action == "gridsearch":
        if not args.use_default_weights and not args.weights:
            args.use_default_weights = True
        stage_gridsearch(args)

    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
