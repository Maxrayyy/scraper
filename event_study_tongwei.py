"""
通威股份（600438.SH）绿色科技创新债券发行公告 —— 事件研究法

事件日:   2025-07-29（第一期绿色科技创新债券发行公告日）
事件窗口: [-5, 5]（公告日前5个交易日至公告日后5个交易日）
估计窗口: [-60, -11]（公告日前60个交易日至公告日前11个交易日）
模型:     市场调整模型  AR_t = R_{i,t} - R_{m,t}
市场基准: 沪深300指数
数据来源: AKShare (东方财富)
"""

import sys
import time
import numpy as np
import pandas as pd
import akshare as ak
from pathlib import Path

# ============================= 参数 =============================
STOCK_CODE   = "600438"           # 通威股份
MARKET_CODE  = "sh000300"         # 沪深300指数 (AKShare格式)
EVENT_DATE   = "2025-07-29"       # 事件日
EVENT_WIN    = (-5, 5)            # 事件窗口
EST_WIN      = (-60, -11)         # 估计窗口
OUTPUT_DIR   = Path("output/event_study")

# ======================== 1. 数据获取 (AKShare) ========================

def _retry_call(func, retries=3, delay=3, **kwargs):
    """带重试的API调用。"""
    for i in range(retries):
        try:
            return func(**kwargs)
        except Exception as e:
            if i < retries - 1:
                print(f"  请求失败: {e}, {delay}秒后重试 ({i+1}/{retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def fetch_from_akshare():
    """通过AKShare获取个股日K线和沪深300指数日K线（真实数据）。"""
    event_dt = pd.Timestamp(EVENT_DATE)
    # 多取日历日以覆盖足够交易日
    start = (event_dt - pd.Timedelta(days=130)).strftime("%Y%m%d")
    end   = (event_dt + pd.Timedelta(days=25)).strftime("%Y%m%d")

    # --- 个股日K线（东方财富，前复权） ---
    print(f"[AKShare] 拉取通威股份({STOCK_CODE}) 日K线 ...")
    stock_df = _retry_call(
        ak.stock_zh_a_hist,
        symbol=STOCK_CODE,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq"   # 前复权
    )
    stock_df = stock_df.rename(columns={"日期": "trade_date", "收盘": "stock_close"})
    stock_df["trade_date"] = pd.to_datetime(stock_df["trade_date"])
    stock_df = stock_df[["trade_date", "stock_close"]]
    print(f"  获取到 {len(stock_df)} 条记录")
    time.sleep(3)

    # --- 沪深300指数日K线 ---
    print(f"[AKShare] 拉取沪深300指数 日K线 ...")
    market_df = _retry_call(ak.stock_zh_index_daily, symbol=MARKET_CODE)
    market_df = market_df.reset_index()
    market_df = market_df.rename(columns={"date": "trade_date", "close": "market_close"})
    market_df["trade_date"] = pd.to_datetime(market_df["trade_date"])
    market_df = market_df[["trade_date", "market_close"]]
    # 按日期范围筛选
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    market_df = market_df[(market_df["trade_date"] >= start_dt) &
                          (market_df["trade_date"] <= end_dt)]
    print(f"  获取到 {len(market_df)} 条记录")

    # --- 合并 ---
    df = pd.merge(stock_df, market_df, on="trade_date", how="inner")
    df = df.sort_values("trade_date").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "raw_price_data.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"原始数据已缓存 -> {path}  ({len(df)} 条)\n")
    return df


def load_data():
    """优先读取本地缓存；否则通过AKShare获取真实数据。"""
    cache = OUTPUT_DIR / "raw_price_data.csv"
    if cache.exists():
        print(f"读取本地缓存: {cache}")
        return pd.read_csv(cache, parse_dates=["trade_date"])

    return fetch_from_akshare()

# ======================== 2. 计算 ========================

def run_event_study(df):
    """核心计算：收益率 → 定位窗口 → AR / CAR → t检验。"""

    # --- 日收益率 ---
    df = df.copy()
    df["Ri"] = df["stock_close"].pct_change()   # 个股收益率
    df["Rm"] = df["market_close"].pct_change()   # 市场收益率
    df.dropna(subset=["Ri", "Rm"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- 定位事件日 ---
    event_dt = pd.Timestamp(EVENT_DATE)
    idx = (df["trade_date"] - event_dt).abs().idxmin()
    print(f"事件日 {EVENT_DATE} -> 匹配交易日 {df.loc[idx, 'trade_date'].date()}"
          f" (索引 {idx})\n")

    # --- 估计窗口 ---
    e0 = max(idx + EST_WIN[0], 0)
    e1 = idx + EST_WIN[1]
    est = df.iloc[e0:e1 + 1].copy()
    est["AR"] = est["Ri"] - est["Rm"]

    sigma = est["AR"].std(ddof=1)           # AR标准差
    mu    = est["AR"].mean()                # AR均值（市场调整模型下理论近似0）
    N_est = len(est)

    print(f"估计窗口 [{EST_WIN[0]}, {EST_WIN[1]}]")
    print(f"  日期: {est['trade_date'].iloc[0].date()} ~ {est['trade_date'].iloc[-1].date()}")
    print(f"  交易日数: {N_est}")
    print(f"  AR均值:   {mu:.6f}")
    print(f"  AR标准差: {sigma:.6f}")

    # --- 事件窗口 ---
    w0 = idx + EVENT_WIN[0]
    w1 = idx + EVENT_WIN[1]
    evt = df.iloc[w0:w1 + 1].copy()
    evt["tau"] = range(EVENT_WIN[0], EVENT_WIN[1] + 1)
    evt["AR"]  = evt["Ri"] - evt["Rm"]
    evt["CAR"] = evt["AR"].cumsum()

    # t统计量
    evt["t_AR"]  = evt["AR"] / sigma
    evt["t_CAR"] = evt["CAR"] / (sigma * np.sqrt(range(1, len(evt) + 1)))

    return evt, sigma


# ======================== 3. 输出 ========================

def print_table(evt):
    """在控制台打印结果表格。"""
    header = (f"{'tau':>4} | {'日期':>12} | {'Ri':>9} | {'Rm':>9} | "
              f"{'AR':>9} | {'CAR':>9} | {'t(AR)':>7} | {'t(CAR)':>7}")
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for _, r in evt.iterrows():
        sig = "**" if abs(r["t_AR"]) > 1.96 else ""
        print(f"{int(r['tau']):>4} | {r['trade_date'].strftime('%Y-%m-%d'):>12} | "
              f"{r['Ri']:>9.4%} | {r['Rm']:>9.4%} | "
              f"{r['AR']:>9.4%} | {r['CAR']:>9.4%} | "
              f"{r['t_AR']:>7.3f} | {r['t_CAR']:>7.3f} {sig}")
    print(sep)
    print("** 表示 |t(AR)| > 1.96 (5%显著性水平)\n")

    # 子窗口CAR
    print("主要子窗口 CAR:")
    for label, a, b in [("[-5,5]",-5,5), ("[-5,-1]",-5,-1),
                         ("[0,0]",0,0), ("[0,5]",0,5),
                         ("[-1,1]",-1,1), ("[-3,3]",-3,3)]:
        sub = evt[(evt["tau"] >= a) & (evt["tau"] <= b)]
        car = sub["AR"].sum()
        n   = len(sub)
        t   = car / (sub["AR"].std(ddof=1) * np.sqrt(n)) if n > 1 else np.nan
        print(f"  CAR{label:>8} = {car:>9.4%}   t = {t:>7.3f}")


def export_excel(evt, sigma):
    """导出Excel（含结果sheet和参数sheet）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = evt[["tau","trade_date","stock_close","market_close",
               "Ri","Rm","AR","CAR","t_AR","t_CAR"]].copy()
    out.columns = ["tau","交易日期","股票收盘价","市场收盘价",
                   "Ri","Rm","AR","CAR","t(AR)","t(CAR)"]

    params = pd.DataFrame({
        "参数": ["股票代码","股票名称","市场基准","事件日",
                "事件窗口","估计窗口","模型","σ(AR)"],
        "值":  [STOCK_CODE,"通威股份",MARKET_CODE,EVENT_DATE,
                f"[{EVENT_WIN[0]},{EVENT_WIN[1]}]",
                f"[{EST_WIN[0]},{EST_WIN[1]}]",
                "市场调整模型 AR=Ri-Rm", f"{sigma:.6f}"]
    })

    path = OUTPUT_DIR / "event_study_results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="AR与CAR", index=False)
        params.to_excel(w, sheet_name="参数设置", index=False)
    print(f"Excel -> {path}")

    csv_path = OUTPUT_DIR / "event_study_results.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV   -> {csv_path}")


def plot_chart(evt):
    """绘制AR柱状图 + CAR折线图。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("matplotlib未安装，跳过绘图。")
        return

    taus = evt["tau"].values
    ar   = evt["AR"].values * 100
    car  = evt["CAR"].values * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in ar]
    ax1.bar(taus, ar, color=colors, edgecolor="grey", linewidth=.5)
    ax1.axhline(0, color="k", lw=.8)
    ax1.axvline(0, color="grey", ls="--", lw=.8, alpha=.5)
    ax1.set_ylabel("AR (%)")
    ax1.set_title(f"通威股份({STOCK_CODE}) 绿色科技创新债券 事件研究\n事件日 {EVENT_DATE}")
    ax1.grid(axis="y", alpha=.3)

    ax2.plot(taus, car, "b-o", ms=5, lw=2)
    ax2.fill_between(taus, 0, car, alpha=.15, color="blue")
    ax2.axhline(0, color="k", lw=.8)
    ax2.axvline(0, color="grey", ls="--", lw=.8, alpha=.5)
    ax2.set_xlabel("相对事件日 (τ)")
    ax2.set_ylabel("CAR (%)")
    ax2.grid(axis="y", alpha=.3)

    plt.xticks(taus)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "event_study_chart.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"图表 -> {fig_path}")
    plt.show()


# ======================== main ========================

def main():
    print("=" * 55)
    print(" 通威股份 绿色科技创新债券 事件研究")
    print("=" * 55)
    print(f" 股票: {STOCK_CODE}  基准: {MARKET_CODE}")
    print(f" 事件日: {EVENT_DATE}")
    print(f" 事件窗口: {list(EVENT_WIN)}  估计窗口: {list(EST_WIN)}")
    print(f" 模型: 市场调整模型\n")

    df = load_data()
    evt, sigma = run_event_study(df)
    print_table(evt)
    export_excel(evt, sigma)
    plot_chart(evt)
    print("\n分析完成。")


if __name__ == "__main__":
    main()
