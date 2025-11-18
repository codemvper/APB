import sys
import argparse
import pandas as pd
import os

try:
    import tushare as ts
except Exception as e:
    print("tushare import error:", e)
    sys.exit(1)

try:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import config
except Exception as e:
    print("config import error:", e)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Inspect raw close, adj_factor, and qfq close for a window")
    parser.add_argument("ts_code", nargs="?", default="513100.SH")
    parser.add_argument("start", nargs="?", default="20220110")
    parser.add_argument("end", nargs="?", default="20220120")
    args = parser.parse_args()

    ts.set_token(config.TUSHARE_TOKEN)
    pro = ts.pro_api()

    px = pro.fund_daily(ts_code=args.ts_code, start_date=args.start, end_date=args.end, fields="ts_code,trade_date,close")
    if px is None or px.empty:
        px = pro.daily(ts_code=args.ts_code, start_date=args.start, end_date=args.end, fields="ts_code,trade_date,close")

    adj = pro.fund_adj(ts_code=args.ts_code, start_date=args.start, end_date=args.end)

    df = pd.merge(px, adj, on=["ts_code", "trade_date"], how="left")
    df["adj_factor"] = df["adj_factor"].astype(float)
    df = df.sort_values("trade_date")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    df.rename(columns={"close": "close_raw"}, inplace=True)
    df["adj_factor"] = df["adj_factor"].ffill().bfill()
    # 计算两种前复权：
    # 前复权（锚定最新因子）：qfq = close * adj_factor / latest_adj_factor
    latest_factor = float(df["adj_factor"].iloc[-1]) if not df.empty else 1.0
    df["close_qfq"] = (df["close_raw"].astype(float) * df["adj_factor"] / latest_factor).round(3)

    out = df[["trade_date", "close_raw", "adj_factor", "close_qfq"]]
    # 展示时统一保留三位小数（不影响内部计算精度）
    disp = out.copy()
    disp["close_raw"] = pd.to_numeric(disp["close_raw"], errors="coerce").round(3)
    disp["adj_factor"] = pd.to_numeric(disp["adj_factor"], errors="coerce").round(3)
    disp["close_qfq"] = pd.to_numeric(disp["close_qfq"], errors="coerce").round(3)
    print(disp.to_string(index=False))

    # highlight 2022-01-12 if within window
    m = out[out["trade_date"] == "2022-01-12"]
    if not m.empty:
        cr = float(m["close_raw"].iloc[0])
        af = float(m["adj_factor"].iloc[0])
        cq = float(m["close_qfq"].iloc[0])
        print(
            "\nHighlight 2022-01-12 ->",
            f" close_raw={cr:.3f}",
            f" adj_factor={af:.3f}",
            f" close_qfq={cq:.3f}",
            f" latest_factor={latest_factor:.3f}",
        )


if __name__ == "__main__":
    main()