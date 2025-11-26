# 全天候组合回测工具

一个面向投资组合的轻量级回测与数据工具，集成“数据获取 → 验证 → 回测 → 图表 → 报告”全流程，默认支持多ETF组合、可配置权重与再平衡频率。

📸 应用截图

 

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

🧠 均线优化策略 (T-Value)
全天候策略支持基于移动平均线（SMA）的动态仓位调整：
```bash
# 使用 T-Value 策略，指定 SMA 窗口（如 30/80/200）
python main.py --action backtest --strategy tvalue --sma50 30 --sma100 80 --sma200 200 --use_default_weights
```

🔍 网格搜索 (Grid Search)
寻找最佳 SMA 参数组合，遍历不同窗口期计算年化、夏普、最大回撤等指标：
```bash
# 运行网格搜索（使用默认遍历列表）
python main.py --action gridsearch --use_default_weights

# 自定义遍历范围（逗号分隔）
python main.py --action gridsearch --use_default_weights --sma50_list "30,40,50" --sma100_list "80,100" --sma200_list "200,250"
```
结果将保存至：`reports/gridsearch_sma.csv`

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
- 代码列表：`--codes 511010 511880 ...`
- 时间范围：`--start YYYY-MM-DD --end YYYY-MM-DD`
- 再平衡频率：`--rebalance NONE|M|Q|A`
- 策略模式：`--strategy fixed|tvalue`（默认 `fixed`）
- SMA参数：`--sma50`, `--sma100`, `--sma200`（仅在 tvalue 模式下生效）
- 网格搜索范围：`--sma50_list`, `--sma100_list`, `--sma200_list`（逗号分隔字符串）
- 自定义权重：`--weights 511010.SH=0.30,...`

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
