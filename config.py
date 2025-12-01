import datetime as _dt

# === 基础配置 ===
# 请按需修改，但默认已包含用户给定权重与token

TUSHARE_TOKEN = ""

# ETF权重配置（合计100%）
ETF_WEIGHTS = {
    "511010.SH": 0.30,  # 长债ETF
    "511880.SH": 0.25,  # 货币ETF
    "510300.SH": 0.15,  # 沪深300
    "513100.SH": 0.15,  # 纳指100(QDII)
    "518880.SH": 0.15,  # 黄金ETF
}

# 动态计算过去12年区间
_today = _dt.date.today()
START_DATE = (_today.replace(year=_today.year - 12)).strftime("%Y-%m-%d")
END_DATE = _today.strftime("%Y-%m-%d")

# 访问频控：随机间隔（秒）以降低被限风险
REQUEST_INTERVAL_MIN_SECONDS = 1.0
REQUEST_INTERVAL_MAX_SECONDS = 3.0

# 价格复权模式：'none' 不复权，'qfq' 前复权，'hfq' 后复权
# 注：ETF在 Tushare 的 pro_bar(asset='E') 对部分品种支持前/后复权；
# 若接口不支持，将回退到原始收盘价（fund_daily / daily）。
ETF_ADJUST_MODE = "qfq"

# 目录
DATA_DIR = "data"
CHART_DIR = "charts"
REPORT_DIR = "reports"
LOG_DIR = "logs"

# 回测参数
# 默认“不再平衡”，可在命令行或此处改为 "M"(月) / "Q"(季) / "A"(年)
REBALANCE_FREQ = "NONE"
RISK_FREE_ANNUAL = 0.0  # 夏普比率风险自由利率（年化），默认0
STRATEGY_MODE = "fixed"

# 验证参数
MAX_ABS_DAILY_RETURN = 0.20  # 单日涨跌幅超过此阈值标记为异常

MAD_THRESHOLD = 5.0  # 基于中位数绝对偏差的异常阈值
