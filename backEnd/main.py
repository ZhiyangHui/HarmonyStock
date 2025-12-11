# ===== 0. 先彻底关掉代理，再导入其他库 =====
import os

# 0.1 清理所有常见代理环境变量
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]:
    os.environ.pop(k, None)

# 0.2 告诉 requests：完全不要管环境里的代理
import requests  # 注意：必须在 akshare 之前
requests.sessions.Session.trust_env = False

import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import akshare as ak
import pandas as pd
import traceback
from datetime import datetime, timedelta
from typing import Literal, Optional, List


# ===== 缓存模块：所有 API 共用 =====
CACHE = {}  # { key: {"data": result, "ts": timestamp} }
CACHE_TTL = 30  # 30 秒缓存


def get_cache(cache_key: str):
    now = time.time()
    item = CACHE.get(cache_key)
    if item and now - item["ts"] < CACHE_TTL:
        return item["data"]
    return None


def set_cache(cache_key: str, data):
    CACHE[cache_key] = {
        "data": data,
        "ts": time.time()
    }


# ===== FastAPI 初始化 =====
app = FastAPI(title="HarmonyStock Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "HarmonyStock backend is running"}


# ===== 通用数据结构 =====
class Quote(BaseModel):
    code: str
    name: str
    price: float
    change: float
    change_percent: float


class KlinePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


# ===== 工具函数 =====

def map_stock_code_to_symbol(code: str) -> str:
    if code[0] in ("6", "9"):
        return f"sh{code}"
    else:
        return f"sz{code}"


def map_index_code_to_symbol(code: str) -> str:
    if code.startswith("399"):
        return f"sz{code}"
    else:
        return f"sh{code}"


def map_stock_code_to_xq_symbol(code: str) -> str:
    if code[0] in ("0", "2", "3"):
        return f"SZ{code}"
    elif code[0] in ("6", "9"):
        return f"SH{code}"
    else:
        raise ValueError(f"invalid stock code: {code}")


# =====================================================================
# 1. 实时行情列表（仅指数）
# =====================================================================

@app.get("/api/quote", response_model=List[Quote])
def get_quote(
    type: str = Query("index"),
    codes: Optional[str] = None
):
    cache_key = f"index_list_{codes}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    try:
        if type != "index":
            raise HTTPException(status_code=400, detail="only index supported")

        df = ak.stock_zh_index_spot_sina()
        if df is None or df.empty:
            raise HTTPException(500, "failed to load index data")

        if codes:
            wanted_codes = [c.strip() for c in codes.split(",")]
        else:
            wanted_codes = ["000001", "399001", "399006"]

        symbols = [map_index_code_to_symbol(c) for c in wanted_codes]
        df_sel = df[df["代码"].isin(symbols)]

        result = []
        for _, row in df_sel.iterrows():
            raw_pct = row["涨跌幅"]
            if isinstance(raw_pct, str):
                raw_pct = raw_pct.replace("%", "")
            result.append(
                Quote(
                    code=row["代码"][-6:],
                    name=row["名称"],
                    price=float(row["最新价"]),
                    change=float(row["涨跌额"]),
                    change_percent=float(raw_pct)
                )
            )

        set_cache(cache_key, result)
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# 2. 单条实时行情（指数 / 股票）
# =====================================================================

@app.get("/api/quote/by_code", response_model=Quote)
def get_quote_by_code(
    type: Literal["index", "stock"],
    code: str,
):

    cache_key = f"quote_{type}_{code}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    try:
        # ===== 指数 =====
        if type == "index":
            df = ak.stock_zh_index_spot_sina()
            if df is None or df.empty:
                raise HTTPException(500, "failed to load index data")

            symbol = map_index_code_to_symbol(code)
            df_sel = df[df["代码"] == symbol]

            if df_sel.empty:
                raise HTTPException(404, "index not found")

            row = df_sel.iloc[0]
            pct = row["涨跌幅"]
            if isinstance(pct, str):
                pct = pct.replace("%", "")

            result = Quote(
                code=code,
                name=row["名称"],
                price=float(row["最新价"]),
                change=float(row["涨跌额"]),
                change_percent=float(pct),
            )

            set_cache(cache_key, result)
            return result

        # ===== 股票（雪球）=====
        xq_symbol = map_stock_code_to_xq_symbol(code)
        df = ak.stock_individual_spot_xq(symbol=xq_symbol)

        if df is None or df.empty:
            raise HTTPException(404, "stock not found")

        row = df.iloc[0]
        result = Quote(
            code=code,
            name=row.get("name", ""),
            price=float(row.get("current_price", 0)),
            change=float(row.get("change_amount", 0)),
            change_percent=float(row.get("change_rate", 0)),
        )

        set_cache(cache_key, result)
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# =====================================================================
# 3. K 线（指数 / 股票）
# =====================================================================

@app.get("/api/kline", response_model=List[KlinePoint])
def get_kline(
    type: Literal["index", "stock"],
    code: str,
    period: Literal["day", "week", "month"] = "day",
    limit: int = 60,
):

    cache_key = f"kline_{type}_{code}_{period}_{limit}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    try:
        if type == "index":
            symbol = map_index_code_to_symbol(code)
            df = ak.stock_zh_index_daily(symbol=symbol)
        else:
            symbol = map_stock_code_to_symbol(code)
            end = datetime.today()
            start = end - timedelta(days=365 * 5)
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )

        if df is None or df.empty:
            raise HTTPException(404, "no kline data")

        # 列名统一
        mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        }
        df = df.rename(columns=mapping)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # 周/月 聚合
        if period == "week":
            df = (
                df.resample("W-FRI", on="date")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                })
                .dropna()
                .reset_index()
            )
        elif period == "month":
            df = (
                df.resample("M", on="date")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                })
                .dropna()
                .reset_index()
            )

        df = df.tail(limit)

        result = [
            KlinePoint(
                date=str(row["date"].date()),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for _, row in df.iterrows()
        ]

        set_cache(cache_key, result)
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


# =====================================================================
# 入口
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
