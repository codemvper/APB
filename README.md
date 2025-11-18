# 全天候组合回测工具

一个面向投资组合的轻量级回测与数据工具，集成“数据获取 → 验证 → 回测 → 图表 → 报告”全流程，默认支持多ETF组合、可配置权重与再平衡频率。

📸 应用截图

> 提示：请将上方占位替换为你的图表截图（`charts/portfolio.html` 导出为图片），或添加动画演示。

✨ 核心功能
- 数据获取与增量更新：Tushare Pro（基金/ETF日线）与复权因子，自动合并去重
- 数据验证：交易日完整性、异常值鲁棒检测、AkShare交叉验证
- 回测引擎：多资产组合的持仓模拟，支持不再平衡/按月(M)/季(Q)/年(A)再平衡
- 可视化与报告：Plotly 交互式净值曲线、最大回撤标注；输出 Markdown 指标报告

🚀 快速开始
📋 环境要求
- Python 3.10+
- Windows / macOS / Linux

⚡ 安装步骤
```bash
git clone <你的仓库地址>
cd 全天候策略
python -m pip install -r requirements.txt
```

🔑 配置 Tushare Token
- 当前默认在 `config.py` 配置 `TUSHARE_TOKEN`（`config.py:6`）。
- 开源时建议移除明文 Token，改为环境变量或运行时注入：
```bash
# Windows（当前会话）
set TUSHARE_TOKEN=your_token_here
# macOS/Linux（当前会话）
export TUSHARE_TOKEN=your_token_here
```

▶️ 运行全流程（获取→验证→回测→图表→报告）
```bash
python main.py --action all --use_default_weights
```

📈 仅回测（基于 data/ 的现有CSV）
```bash
python main.py --action backtest --use_default_weights --rebalance M
```

⏱️ 获取/更新/验证
```bash
# 全量获取数据
python main.py --action fetch --start 2013-01-01 --end 2025-11-17
# 增量更新（按CSV最后交易日的下一天补到 --end，未指定则为今天）
python main.py --action update --end 2025-11-17
# 仅验证数据
python main.py --action validate
```

🎛️ 可选参数
- 代码列表：`--codes 511010 511880 510300 513100 518880`（无后缀自动补齐，`utils.py:42–51`）
- 时间范围：`--start YYYY-MM-DD --end YYYY-MM-DD`
- 再平衡频率：`--rebalance NONE|M|Q|A`（默认 `NONE`，`config.py:37–41`）
- 自定义权重：`--weights 511010.SH=0.30,511880.SH=0.25,...`（总和建议为 1.0）

📂 项目结构
```
全天候策略/
├── main.py                 # 流水线入口与参数解析
├── data_fetcher.py         # 数据抓取/复权/CSV写入
├── validator.py            # 完整性检查/异常检测/交叉验证
├── backtest.py             # 组合回测与再平衡模拟
├── visualization.py        # 可视化与最大回撤标注
├── report.py               # 指标计算与Markdown报告
├── config.py               # 参数与目录配置
├── requirements.txt        # 依赖清单
├── scripts/inspect_adj.py  # 复权因子与价格窗口检查脚本
├── data/                   # CSV数据输出（运行生成）
├── charts/                 # 图表HTML输出（运行生成）
├── reports/                # 报告Markdown输出（运行生成）
└── logs/                   # 日志文件（运行生成）
```

🧠 关键实现
- CSV 命名与格式：`<ts_code.replace('.', '_')>.csv`；列：`交易日期, ETF代码, 收盘价`（`data_fetcher.py:111–124`）。写入会与历史合并去重，收盘价保留三位小数（`data_fetcher.py:128–137`）。
- 复权口径：`fund_adj` 因子合并当日收盘价；前复权（`ETF_ADJUST_MODE='qfq'`）采用“锚定区间最新因子”标准：`qfq = close * adj_factor / latest_adj_factor`（`data_fetcher.py:58–63, 93–101`）。
- 回测与再平衡：将资产收盘价对齐为共同索引，持仓价值按设定频率回到目标权重（`backtest.py:8–21, 23–70`）。
- 指标与报告：年化收益率、波动率、夏普、最大回撤等（`report.py:8–36`），并保存到 `reports/backtest_report.md`（`report.py:53–66`）。

📤 输出位置
- CSV：`data/`
- 图表：`charts/portfolio.html`
- 报告：`reports/backtest_report.md`
- 日志：`logs/app.log`

🧩 故障排除
- `python` 在 Windows 无输出或命中 Microsoft Store 别名：关闭“设置 → 应用 → 应用执行别名”中的 `python.exe/python3.exe`，或使用绝对路径执行。
- 依赖安装失败：升级 pip 或使用镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple/`
- Tushare 调用失败：检查 Token 是否有效；网络是否可用；适当降低调用频率。

🤝 贡献与许可
- 欢迎提交 Issue/PR 进行改进与功能扩展。
- 建议采用 MIT 许可证并添加 `LICENSE` 文件。


- 写入时与历史数据合并并去重，不会覆盖已存在记录。

如担心限流，可在 `config.py` 调整随机间隔：
- `REQUEST_INTERVAL_MIN_SECONDS = 1.0`
- `REQUEST_INTERVAL_MAX_SECONDS = 3.0`

价格与复权：
- 采用 `fund_adj` 的复权因子 `adj_factor` 与当日收盘价合并，前复权采用“锚定最新因子”的标准口径：`qfq = close * adj_factor / latest_adj_factor`。
- 最新因子取所请求区间的最后一个交易日的 `adj_factor`；这样最新日价格保持不变、历史价格按比例缩放以消除除权/拆分跳变。
- 当日收盘价优先来自 `fund_daily`，如为空则回退到 `daily`；复权因子来自 `fund_adj`（每日），缺失时前向/后向填充。
- 在 `config.py` 将 `ETF_ADJUST_MODE` 设为 `qfq` 时启用上述前复权；设为 `none` 时保存原始收盘价。

只验证数据：
```bash
python main.py --action validate
```

启用月度再平衡：
```bash
python main.py --action backtest --use_default_weights --rebalance M
```

只回测并生成图表与报告：
```bash
python main.py --action backtest --use_default_weights
```

## 输出位置
- CSV数据：`data/`
- 图表HTML：`charts/portfolio.html`
- 回测报告：`reports/backtest_report.md`
- 日志：`logs/app.log`

## 说明与限制
- AkShare交叉验证需网络与可访问东财接口，失败时不会影响主流程。
- 夏普比率使用年化风险自由利率（默认0），可在 `config.py` 中调整。
- 回测仅使用 `data/` 目录下的 CSV 数据（`main.py` 中的 `load_csv_close` 从该目录读取）。
- 默认不再平衡（`NONE`），权重自然随收益漂移；如需定期再平衡，可设置为月度 `M`、季度 `Q` 或年度 `A`。