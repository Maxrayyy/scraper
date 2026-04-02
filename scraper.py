"""
光伏行业30家上市公司季度财务数据爬取脚本

数据来源: 东方财富 (通过AKShare)
数据范围: 2019Q1 ~ 2025Q3
指标: 净利润、总资产、总负债、营业收入
"""

import time
import logging
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 公司配置: (股票代码, 公司简称, AKShare symbol前缀)
# 上交所以6开头 -> SH, 深交所以0/3开头 -> SZ
# ---------------------------------------------------------------------------
COMPANIES = [
    ("601012", "隆基绿能"),
    ("600438", "通威股份"),
    ("688223", "晶科能源"),
    ("688599", "天合光能"),
    ("002459", "晶澳科技"),
    ("002129", "TCL中环"),
    ("300274", "阳光电源"),
    ("688303", "大全能源"),
    ("603806", "福斯特"),
    ("601865", "福莱特"),
    ("002506", "协鑫集成"),
    ("300118", "东方日升"),
    ("300393", "中来股份"),
    ("600732", "爱旭股份"),
    ("688680", "海优新材"),
    ("603212", "赛伍技术"),
    ("300842", "帝科股份"),
    ("002079", "苏州固锝"),
    ("603396", "金辰股份"),
    ("300724", "捷佳伟创"),
    ("600537", "亿晶光电"),
    ("002218", "拓日新能"),
    ("600151", "航天机电"),
    ("600481", "双良节能"),
    ("603185", "上机数控"),
    ("601908", "京运通"),
    ("300700", "岱勒新材"),
    ("002132", "恒星科技"),
    ("300041", "回天新材"),
    ("301040", "中环海陆"),
]

# 季度报告截止日（月-日）
QUARTER_ENDINGS = {"03-31", "06-30", "09-30", "12-31"}
DATE_START = pd.Timestamp("2019-01-01")
DATE_END = pd.Timestamp("2025-09-30")

# 重试配置
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # 秒，指数退避基数


def make_symbol(code: str) -> str:
    """根据股票代码生成AKShare所需的symbol格式"""
    if code.startswith("6"):
        return f"SH{code}"
    else:
        return f"SZ{code}"


def fetch_with_retry(func, symbol: str, retries: int = MAX_RETRIES) -> pd.DataFrame:
    """带重试机制的API调用"""
    for attempt in range(retries):
        try:
            df = func(symbol=symbol)
            return df
        except Exception as e:
            wait = RETRY_BACKOFF ** (attempt + 1)
            if attempt < retries - 1:
                logger.warning(
                    f"请求失败 ({symbol}): {e}, {wait}秒后重试 ({attempt+1}/{retries})"
                )
                time.sleep(wait)
            else:
                logger.error(f"请求最终失败 ({symbol}): {e}")
                return pd.DataFrame()


def quarter_label(date: pd.Timestamp) -> str:
    """将日期转换为季度标签，如 2024Q1"""
    month = date.month
    quarter_map = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}
    q = quarter_map.get(month, "")
    return f"{date.year}{q}"


def fetch_company_data(code: str, name: str) -> pd.DataFrame:
    """爬取单个公司的财务数据"""
    symbol = make_symbol(code)

    # 获取资产负债表
    balance_df = fetch_with_retry(ak.stock_balance_sheet_by_report_em, symbol)
    time.sleep(1)

    # 获取利润表
    profit_df = fetch_with_retry(ak.stock_profit_sheet_by_report_em, symbol)
    time.sleep(1)

    if balance_df.empty and profit_df.empty:
        logger.warning(f"{name}({code}): 资产负债表和利润表均无数据")
        return pd.DataFrame()

    rows = []

    # 处理资产负债表
    balance_data = {}
    if not balance_df.empty and "REPORT_DATE" in balance_df.columns:
        for _, row in balance_df.iterrows():
            rd = pd.Timestamp(row["REPORT_DATE"])
            date_str = rd.strftime("%m-%d")
            if date_str in QUARTER_ENDINGS and DATE_START <= rd <= DATE_END:
                balance_data[rd] = {
                    "total_assets": row.get("TOTAL_ASSETS"),
                    "total_liabilities": row.get("TOTAL_LIABILITIES"),
                }

    # 处理利润表
    profit_data = {}
    if not profit_df.empty and "REPORT_DATE" in profit_df.columns:
        for _, row in profit_df.iterrows():
            rd = pd.Timestamp(row["REPORT_DATE"])
            date_str = rd.strftime("%m-%d")
            if date_str in QUARTER_ENDINGS and DATE_START <= rd <= DATE_END:
                profit_data[rd] = {
                    "net_profit": row.get("PARENT_NETPROFIT", row.get("NETPROFIT")),
                    "revenue": row.get("OPERATE_INCOME", row.get("TOTAL_OPERATE_INCOME")),
                }

    # 合并
    all_dates = sorted(set(balance_data.keys()) | set(profit_data.keys()))
    for rd in all_dates:
        entry = {
            "stock_code": code,
            "stock_name": name,
            "report_date": rd.strftime("%Y-%m-%d"),
            "quarter": quarter_label(rd),
        }
        if rd in balance_data:
            entry.update(balance_data[rd])
        if rd in profit_data:
            entry.update(profit_data[rd])
        rows.append(entry)

    return pd.DataFrame(rows)


def main():
    logger.info(f"开始爬取 {len(COMPANIES)} 家公司的财务数据...")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    all_data = []
    failed = []

    for code, name in tqdm(COMPANIES, desc="爬取进度"):
        logger.info(f"正在处理: {name} ({code})")
        try:
            df = fetch_company_data(code, name)
            if not df.empty:
                all_data.append(df)
                logger.info(f"  {name}: 获取 {len(df)} 条季度数据")
            else:
                failed.append((code, name))
        except Exception as e:
            logger.error(f"  {name}({code}) 处理异常: {e}")
            failed.append((code, name))

    if not all_data:
        logger.error("未获取到任何数据!")
        return

    # 合并所有数据
    result = pd.concat(all_data, ignore_index=True)

    # 定义列顺序
    columns = [
        "stock_code", "stock_name", "report_date", "quarter",
        "net_profit", "total_assets", "total_liabilities", "revenue",
    ]
    for col in columns:
        if col not in result.columns:
            result[col] = None
    result = result[columns]

    # 排序
    result = result.sort_values(["stock_code", "report_date"]).reset_index(drop=True)

    # 导出Excel
    excel_path = output_dir / "financial_data.xlsx"
    result.to_excel(excel_path, index=False, engine="openpyxl")
    logger.info(f"Excel文件已导出: {excel_path}")

    # 导出CSV (UTF-8 with BOM)
    csv_path = output_dir / "financial_data.csv"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"CSV文件已导出: {csv_path}")

    # 统计汇总
    logger.info(f"\n===== 爬取完成 =====")
    logger.info(f"成功: {len(COMPANIES) - len(failed)} 家公司")
    logger.info(f"总计: {len(result)} 条记录")
    if failed:
        logger.warning(f"失败: {len(failed)} 家公司 -> {[f'{n}({c})' for c, n in failed]}")


if __name__ == "__main__":
    main()