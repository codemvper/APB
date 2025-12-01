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


def _simulate_rebalanced_portfolio(prices: pd.DataFrame, weights: Dict[str, float], freq: str = "M") -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    start_val = 1.0
    norm_freq = _normalize_freq(freq)
    rebal_dates = prices.resample(norm_freq).first().index if norm_freq else pd.DatetimeIndex([])

    holding_value = {c: start_val * weights.get(c, 0.0) for c in prices.columns}

    portfolio_values = []
    per_asset_values = []
    events_rows = []
    prev_prices = prices.iloc[0]

    for dt, row in prices.iterrows():
        if norm_freq and dt in rebal_dates:
            total_val = sum(holding_value.values())
            for c in prices.columns:
                holding_value[c] = total_val * weights.get(c, 0.0)
            for c in prices.columns:
                events_rows.append({
                    "date": dt,
                    "event": "fixed_rebalance",
                    "asset": c,
                    "new_weight": float(weights.get(c, 0.0)),
                    "factor": 1.0,
                    "reason": str(norm_freq)
                })

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
    events_df = pd.DataFrame(events_rows)
    return pf, daily_ret, asset_df, events_df


def _simulate_tvalue_portfolio(prices: pd.DataFrame, weights: Dict[str, float], sma_short: int = 50, sma_mid: int = 100, sma_long: int = 200, confirm_days: int = 5, cooldown_days: int = 10) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    start_val = 1.0
    codes = list(prices.columns)
    cash_code = None
    bond_code = None
    for c in codes:
        if "511880" in c:
            cash_code = c
        if "511010" in c:
            bond_code = c
    equity_like = [c for c in codes if c not in {cash_code, bond_code}]

    holding_value = {c: start_val * weights.get(c, 0.0) for c in prices.columns}
    portfolio_values = []
    per_asset_values = []
    events_rows = []

    factors = {c: 1.0 for c in equity_like}
    last_change = {c: None for c in equity_like}

    sma50 = prices[equity_like].rolling(int(sma_short)).mean()
    sma100 = prices[equity_like].rolling(int(sma_mid)).mean()
    sma200 = prices[equity_like].rolling(int(sma_long)).mean()
    # 增加：计算20日滚动最高价，用于检测高位回撤（V型顶）
    roll_max20 = prices[equity_like].rolling(20).max()
    ret10 = prices[equity_like] / prices[equity_like].shift(10) - 1.0

    tier_map = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}

    def tier_from_factor(f):
        if f >= 1.5:
            return 3
        if f >= 0.75:
            return 2
        if f > 0.0:
            return 1
        return 0

    prev_prices = prices.iloc[0]

    for dt, row in prices.iterrows():
        changed = False
        asset_reason = {}
        asset_prev_tier = {}
        asset_new_tier = {}
        for c in equity_like:
            p = row[c]
            s50 = sma50.loc[dt, c]
            s100 = sma100.loc[dt, c]
            s200 = sma200.loc[dt, c]
            if np.isnan(s50) or np.isnan(s100) or np.isnan(s200):
                continue
            t_val = int((p > s50)) + int((p > s100)) + int((p > s200))
            target_tier = t_val
            conf = False
            pos = prices.index.get_loc(dt)
            if isinstance(pos, int) and pos >= int(confirm_days) - 1:
                idxs = prices.index[pos - (int(confirm_days) - 1) : pos + 1]
                vals = []
                for ix in idxs:
                    pv = int((prices.loc[ix, c] > sma50.loc[ix, c])) + int((prices.loc[ix, c] > sma100.loc[ix, c])) + int((prices.loc[ix, c] > sma200.loc[ix, c]))
                    vals.append(pv)
                if len(set(vals)) == 1 and vals[-1] == target_tier:
                    conf = True

            cooldown = False
            if last_change[c] is not None:
                cooldown = (dt - last_change[c]).days < int(cooldown_days)

            desired_factor = tier_map.get(target_tier, 1.0)
            cur_factor = factors[c]
            cur_tier = tier_from_factor(cur_factor)

            # === 紧急熔断机制：高位回撤检测 (V型顶保护) ===
            # 逻辑：如果当前价格较过去20日最高价下跌超过 5%，且当前仓位较重(Tier>=2)，强制减仓
            # 特点：无视 cooldown，无视均线支撑，优先逃命
            rmax = roll_max20.loc[dt, c]
            is_emergency_cut = False
            if not np.isnan(rmax) and rmax > 0:
                dd_from_peak = (p / rmax) - 1.0
                if dd_from_peak <= -0.05 and cur_tier >= 2:
                    # 强制降级到 Tier 1 (0.5倍)，保住大部分利润
                    new_tier = 1
                    new_factor = tier_map[new_tier]
                    if new_factor != cur_factor:
                        factors[c] = new_factor
                        last_change[c] = dt
                        changed = True
                        asset_reason[c] = f"emergency_cut_dd{dd_from_peak:.1%}"
                        asset_prev_tier[c] = cur_tier
                        asset_new_tier[c] = new_tier
                        is_emergency_cut = True
            
            if is_emergency_cut:
                continue # 已触发熔断，跳过后续普通逻辑

            # 普通逻辑
            if not cooldown and (target_tier < cur_tier) and desired_factor != cur_factor:
                new_tier = max(target_tier, cur_tier - 1)
                new_factor = tier_map[new_tier]
                factors[c] = new_factor
                last_change[c] = dt
                changed = True
                asset_reason[c] = "down_cross"
                asset_prev_tier[c] = cur_tier
                asset_new_tier[c] = new_tier
            elif not cooldown and conf and desired_factor != cur_factor:
                factors[c] = desired_factor
                last_change[c] = dt
                changed = True
                asset_reason[c] = "confirm"
                asset_prev_tier[c] = cur_tier
                asset_new_tier[c] = target_tier
            else:
                r10 = ret10.loc[dt, c]
                if not cooldown and not np.isnan(r10):
                    if r10 >= 0.06 and cur_tier < 3:
                        new_tier = min(3, cur_tier + 1)
                        new_factor = tier_map[new_tier]
                        if new_factor != cur_factor:
                            factors[c] = new_factor
                            last_change[c] = dt
                            changed = True
                            asset_reason[c] = "fast_up"
                            asset_prev_tier[c] = cur_tier
                            asset_new_tier[c] = new_tier
                    elif r10 <= -0.06 and cur_tier > 0:
                        new_tier = max(0, cur_tier - 1)
                        new_factor = tier_map[new_tier]
                        if new_factor != cur_factor:
                            factors[c] = new_factor
                            last_change[c] = dt
                            changed = True
                            asset_reason[c] = "fast_down"
                            asset_prev_tier[c] = cur_tier
                            asset_new_tier[c] = new_tier

        if changed:
            total_val = sum(holding_value.values())
            eq_new = {c: weights.get(c, 0.0) * factors[c] for c in equity_like}
            sum_eq = float(sum(eq_new.values()))
            w_cash = weights.get(cash_code, 0.0) if cash_code else 0.0
            w_bond = weights.get(bond_code, 0.0) if bond_code else 0.0
            base_eq_sum = float(sum(weights.get(c, 0.0) for c in equity_like))
            delta = sum_eq - base_eq_sum
            target_cb = max(0.0, 1.0 - sum_eq)
            if delta >= 0.0:
                reduce_cash = min(w_cash, delta)
                cash_new = max(0.0, w_cash - reduce_cash)
                bond_new = max(0.0, target_cb - cash_new)
            else:
                release = -delta
                cash_new = min(target_cb, w_cash + release)
                bond_new = max(0.0, target_cb - cash_new)

            new_w = {c: 0.0 for c in prices.columns}
            for k, v in eq_new.items():
                new_w[k] = v
            if cash_code:
                new_w[cash_code] = cash_new
            if bond_code:
                new_w[bond_code] = bond_new
            sumb = float(sum(new_w.values()))
            if sumb > 0:
                for k in new_w:
                    new_w[k] = new_w[k] / sumb
            for c in prices.columns:
                holding_value[c] = total_val * new_w.get(c, 0.0)
            for c in prices.columns:
                s50 = sma50.loc[dt, c] if c in sma50.columns else np.nan
                s100 = sma100.loc[dt, c] if c in sma100.columns else np.nan
                s200 = sma200.loc[dt, c] if c in sma200.columns else np.nan
                r10 = ret10.loc[dt, c] if c in ret10.columns else np.nan
                events_rows.append({
                    "date": dt,
                    "event": "tvalue_rebalance",
                    "asset": c,
                    "new_weight": float(new_w.get(c, 0.0)),
                    "factor": float(factors.get(c, 1.0)) if c in equity_like else 1.0,
                    "reason": asset_reason.get(c, ""),
                    "prev_tier": int(asset_prev_tier.get(c, tier_from_factor(factors.get(c, 1.0)))),
                    "new_tier": int(asset_new_tier.get(c, tier_from_factor(factors.get(c, 1.0)))),
                    "price": float(row[c]),
                    "sma50": float(s50) if not np.isnan(s50) else None,
                    "sma100": float(s100) if not np.isnan(s100) else None,
                    "sma200": float(s200) if not np.isnan(s200) else None,
                    "ret10": float(r10) if not np.isnan(r10) else None,
                    "cooldown": bool(last_change.get(c) is not None and (dt - last_change[c]).days < 10)
                })

        for c in prices.columns:
            prev = prev_prices[c]
            if prev == 0 or np.isnan(prev):
                ret = 0.0
            else:
                ret = (row[c] - prev) / prev
            holding_value[c] *= (1.0 + (ret if not np.isnan(ret) else 0.0))

        total_val = sum(holding_value.values())
        portfolio_values.append((dt, total_val))
        per_asset_values.append((dt, {c: holding_value[c] for c in prices.columns}))
        prev_prices = row

    pf = pd.Series({dt: v for dt, v in portfolio_values}).sort_index()
    asset_df = pd.DataFrame({dt: vals for dt, vals in per_asset_values}).T
    asset_df.index = pf.index
    daily_ret = pf.pct_change().fillna(0.0)
    events_df = pd.DataFrame(events_rows)
    return pf, daily_ret, asset_df, events_df


def _simulate_momentum_portfolio(prices: pd.DataFrame, weights: Dict[str, float], momentum_window: int = 20, freq: str = "M") -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    绝对动量策略：
    1. 每月月初（freq="M"）检查一次
    2. 计算过去N个月（近似 momentum_window * 20个交易日）的收益率
    3. 若收益率 > 0，保持原权重
    4. 若收益率 <= 0，清仓该资产，资金按 6:5 分配给 长债(511010) 和 货币(511880)
    """
    start_val = 1.0
    
    # 识别资产类型
    cash_code = None
    bond_code = None
    for c in prices.columns:
        if "511880" in c:
            cash_code = c
        if "511010" in c:
            bond_code = c
    
    # 风险资产池（排除长债和现金）
    risk_assets = [c for c in prices.columns if c not in {cash_code, bond_code}]
    
    # 初始化持仓
    holding_value = {c: start_val * weights.get(c, 0.0) for c in prices.columns}
    
    portfolio_values = []
    per_asset_values = []
    events_rows = []
    
    # 再平衡日期（每月月初/第一个交易日）
    norm_freq = _normalize_freq(freq)
    rebal_dates = set(prices.resample(norm_freq).first().index) if norm_freq else set()
    
    # 转换窗口天数：momentum_window (月) -> 交易日 (约 * 20)
    # 假设输入 momentum_window 是月份数，这里转换为交易日
    lookback_days = int(momentum_window * 20)
    
    prev_prices = prices.iloc[0]
    
    # 记录每个资产当前的“目标状态”：True=持有风险，False=避险
    # 初始默认都持有
    asset_status = {c: True for c in risk_assets}
    
    for dt, row in prices.iterrows():
        # 1. 检查是否是调仓日
        if dt in rebal_dates:
            # 计算总资产
            total_val = sum(holding_value.values())
            
            # 计算新的目标权重
            target_weights = {}
            
            # 基础权重分配（避险资产先拿自己的基础份额）
            w_bond_base = weights.get(bond_code, 0.0) if bond_code else 0.0
            w_cash_base = weights.get(cash_code, 0.0) if cash_code else 0.0
            
            # 累加来自负动量资产的转移权重
            w_bond_extra = 0.0
            w_cash_extra = 0.0
            
            current_idx = prices.index.get_loc(dt)
            
            for c in risk_assets:
                w_base = weights.get(c, 0.0)
                
                # 计算动量：过去N个月收益率
                # 需判断历史数据是否足够
                momentum_val = 0.0
                has_history = False
                
                if current_idx >= lookback_days:
                    past_price = prices[c].iloc[current_idx - lookback_days]
                    curr_price = row[c]
                    if past_price > 0:
                        momentum_val = (curr_price / past_price) - 1.0
                        has_history = True
                
                # 动量判断
                # 如果数据不足，默认持有（或者默认避险？一般默认持有跟上大盘）
                is_positive = True
                if has_history:
                    is_positive = momentum_val > 0
                
                asset_status[c] = is_positive
                
                if is_positive:
                    # 动量为正，维持原权重
                    target_weights[c] = w_base
                else:
                    # 动量为负，清仓，权重分给债/现
                    target_weights[c] = 0.0
                    # 比例 6:5 -> 债 6/11, 现 5/11
                    w_bond_extra += w_base * (6.0 / 11.0)
                    w_cash_extra += w_base * (5.0 / 11.0)
                    
                    events_rows.append({
                        "date": dt,
                        "event": "momentum_cut",
                        "asset": c,
                        "momentum_ret": float(momentum_val),
                        "lookback": lookback_days
                    })
            
            # 分配避险资产权重
            if bond_code:
                target_weights[bond_code] = w_bond_base + w_bond_extra
            elif cash_code:
                # 如果没有长债ETF，全给现金
                target_weights[cash_code] = w_cash_base + w_cash_extra + w_bond_extra # 全给现金
            
            if cash_code:
                target_weights[cash_code] = target_weights.get(cash_code, 0.0) + w_cash_base + w_cash_extra
            elif bond_code:
                 # 如果没有现金ETF，全给长债
                target_weights[bond_code] = target_weights.get(bond_code, 0.0) + w_cash_extra

            # 执行调仓：更新 holding_value
            # 归一化检查（理论上应该和为1）
            sum_w = sum(target_weights.values())
            if sum_w > 0:
                for k in target_weights:
                    target_weights[k] /= sum_w
            
            for c in prices.columns:
                holding_value[c] = total_val * target_weights.get(c, 0.0)

        # 2. 计算日内净值变化
        for c in prices.columns:
            prev = prev_prices[c]
            if prev == 0 or np.isnan(prev):
                ret = 0.0
            else:
                ret = (row[c] - prev) / prev
            holding_value[c] *= (1.0 + (ret if not np.isnan(ret) else 0.0))
            
        total_val = sum(holding_value.values())
        portfolio_values.append((dt, total_val))
        per_asset_values.append((dt, {c: holding_value[c] for c in prices.columns}))
        prev_prices = row

    pf = pd.Series({dt: v for dt, v in portfolio_values}).sort_index()
    asset_df = pd.DataFrame({dt: vals for dt, vals in per_asset_values}).T
    asset_df.index = pf.index
    daily_ret = pf.pct_change().fillna(0.0)
    events_df = pd.DataFrame(events_rows)
    return pf, daily_ret, asset_df, events_df


def backtest(prices_map: Dict[str, pd.DataFrame], weights: Dict[str, float], start_date: str = None, end_date: str = None, freq: str = "M", strategy: str = "fixed", sma_short: int = 50, sma_mid: int = 100, sma_long: int = 200, confirm_days: int = 5, cooldown_days: int = 10, momentum_window: int = 10):
    logger = get_logger("backtest")
    prices = _prepare_price_frame(prices_map)
    if start_date:
        prices = prices.loc[pd.to_datetime(start_date):]
    if end_date:
        prices = prices.loc[:pd.to_datetime(end_date)]
    prices = prices.dropna(how="all")
    logger.info(f"Price frame prepared: {prices.index.min()} -> {prices.index.max()} | {list(prices.columns)}")

    if str(strategy).lower() == "tvalue":
        pf, daily_ret, asset_val, events = _simulate_tvalue_portfolio(prices, weights, sma_short=sma_short, sma_mid=sma_mid, sma_long=sma_long, confirm_days=confirm_days, cooldown_days=cooldown_days)
    elif str(strategy).lower() == "momentum":
        pf, daily_ret, asset_val, events = _simulate_momentum_portfolio(prices, weights, momentum_window=momentum_window, freq=freq)
    else:
        pf, daily_ret, asset_val, events = _simulate_rebalanced_portfolio(prices, weights, freq=freq)
    return pf, daily_ret, asset_val, prices, events


def max_drawdown(series: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    cummax = series.cummax()
    drawdown = series / cummax - 1.0
    mdd = drawdown.min()
    end = drawdown.idxmin()
    start = series.loc[:end].idxmax()
    return float(mdd), start, end