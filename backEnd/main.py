# ===== 0. 先彻底关掉代理，再导入其他库 =====
import os

# 0.1 清理所有常见代理环境变量
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]:
    os.environ.pop(k, None)

# 0.2 禁用 requests 对代理的读取
import requests
requests.sessions.Session.trust_env = False


# ===== 标准库 & 第三方 =====
import time
import traceback
import pandas as pd
from datetime import datetime, timedelta
from typing import Literal, Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import akshare as ak

# ===== 导入数据库模块 =====
from db import SessionLocal, engine, Base
from models import IndexQuoteModel, StockQuoteModel, KlineModel


# ===== 创建数据库表 =====
Base.metadata.create_all(bind=engine)


# ===== 缓存模块 =====
CACHE: dict = {}
CACHE_TTL = 30  # 秒

# ===== 上游重数据源缓存（避免重复昂贵操作） =====

INDEX_SPOT_CACHE = {
    "data": None,
    "ts": 0,
}

STOCK_SPOT_CACHE = {
    "data": None,
    "ts": 0,
}

# 缓存时间（秒）
INDEX_SPOT_TTL = 10   # 指数：轻接口，10 秒足够
STOCK_SPOT_TTL = 10   # 股票：重接口，必须缓存


def get_index_spot_df():
    now = time.time()
    if INDEX_SPOT_CACHE["data"] is not None and now - INDEX_SPOT_CACHE["ts"] < INDEX_SPOT_TTL:
        return INDEX_SPOT_CACHE["data"]

    df = ak.stock_zh_index_spot_sina()
    if df is None or df.empty:
        raise Exception("index spot source error")

    INDEX_SPOT_CACHE["data"] = df
    INDEX_SPOT_CACHE["ts"] = now
    return df


def get_stock_spot_df():
    now = time.time()
    if STOCK_SPOT_CACHE["data"] is not None and now - STOCK_SPOT_CACHE["ts"] < STOCK_SPOT_TTL:
        return STOCK_SPOT_CACHE["data"]

    df = ak.stock_zh_a_spot()
    if df is None or df.empty:
        raise Exception("stock spot source error")

    STOCK_SPOT_CACHE["data"] = df
    STOCK_SPOT_CACHE["ts"] = now
    return df




def get_cache(key: str):
    item = CACHE.get(key)
    if item and time.time() - item["ts"] < CACHE_TTL:
        return item["data"]
    return None


def set_cache(key: str, data):
    CACHE[key] = {"data": data, "ts": time.time()}


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


# ===== Pydantic 数据结构 =====
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
    return f"sh{code}" if code[0] in ("6", "9") else f"sz{code}"


def map_index_code_to_symbol(code: str) -> str:
    return f"sz{code}" if code.startswith("399") else f"sh{code}"


def map_stock_code_to_xq_symbol(code: str) -> str:
    if code[0] in ("0", "2", "3"):
        return f"SZ{code}"
    if code[0] in ("6", "9"):
        return f"SH{code}"
    raise ValueError(f"invalid code {code}")


# =====================================================================
# 1. 实时指数列表接口（带缓存 + 写 DB + DB 兜底）
# =====================================================================
@app.get("/api/quote", response_model=List[Quote])
def get_quote(
    type: str = Query("index"),
    codes: Optional[str] = None
):
    """
    实时行情列表：
    - type = index：预设 3 个指数（或前端传 codes）
    - type = stock：预设 3 只股票（或前端传 codes）
    """
    cache_key = f"quote_list_{type}_{codes}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    # ====== 公共 DB 会话 ======
    db = SessionLocal()
    try:
        # ============================================================
        # 1) 指数列表：新浪 stock_zh_index_spot_sina（基本不变）
        # ============================================================
        if type == "index":
            df = get_index_spot_df()
            if df is None or df.empty:
                raise Exception("index source error")

            if codes:
                codes_list = [c.strip() for c in codes.split(",") if c.strip()]
            else:
                # 默认 3 个：上证 / 深成 / 创业板
                codes_list = ["000001", "399001", "399006"]

            symbols = [map_index_code_to_symbol(c) for c in codes_list]
            df_sel = df[df["代码"].isin(symbols)]

            if df_sel.empty:
                raise Exception("no index rows after filter")

            result: List[Quote] = []

            for _, row in df_sel.iterrows():
                pct = row["涨跌幅"]
                if isinstance(pct, str):
                    pct = pct.replace("%", "")

                q = Quote(
                    code=row["代码"][-6:],
                    name=row["名称"],
                    price=float(row["最新价"]),
                    change=float(row["涨跌额"]),
                    change_percent=float(pct),
                )
                result.append(q)

                db_item = IndexQuoteModel(
                    code=q.code,
                    name=q.name,
                    price=q.price,
                    change=q.change,
                    change_percent=q.change_percent,
                    updated_at=datetime.utcnow(),
                )
                db.merge(db_item)

            db.commit()
            set_cache(cache_key, result)
            return result

        # ============================================================
        # 2) 股票列表：雪球 stock_individual_spot_xq（循环 3 只预设）
        # ============================================================
        elif type == "stock":
            if codes:
                stock_codes = [c.strip() for c in codes.split(",") if c.strip()]
            else:
                stock_codes = ["300750", "601012", "688981"]

            # 关键：一次性取全市场实时行情（避免雪球单票接口）
            df = get_stock_spot_df()

            if df is None or df.empty:
                raise Exception("stock spot source error")

            # 统一列名（不同源列名可能略有差异，你先用这一套；如不匹配我再按你df.columns调整）
            # 期望列：代码 / 名称 / 最新价 / 涨跌额 / 涨跌幅
            needed = ["代码", "名称", "最新价", "涨跌额", "涨跌幅"]
            for col in needed:
                if col not in df.columns:
                    raise Exception(f"unexpected stock spot columns, missing: {col}, got: {list(df.columns)}")

            df_sel = df[df["代码"].isin(stock_codes)]
            if df_sel.empty:
                raise Exception("no stock rows after filter")

            result: List[Quote] = []
            for _, row in df_sel.iterrows():
                pct = row["涨跌幅"]
                if isinstance(pct, str):
                    pct = pct.replace("%", "").strip()

                q = Quote(
                    code=str(row["代码"]),
                    name=str(row["名称"]),
                    price=float(row["最新价"]),
                    change=float(row["涨跌额"]),
                    change_percent=float(pct),
                )
                result.append(q)

                db.merge(StockQuoteModel(
                    code=q.code,
                    name=q.name,
                    price=q.price,
                    change=q.change,
                    change_percent=q.change_percent,
                    updated_at=datetime.utcnow(),
                ))

            db.commit()
            set_cache(cache_key, result)
            return result

        # ============================================================
        # 3) 其它 type 不支持
        # ============================================================
        else:
            raise HTTPException(status_code=400, detail="only index or stock supported")

    except Exception:
        traceback.print_exc()

        # ====== 兜底：数据库读取 ======
        if type == "index":
            items = db.query(IndexQuoteModel).all()
        elif type == "stock":
            items = db.query(StockQuoteModel).all()
        else:
            items = []

        if not items:
            db.close()
            raise HTTPException(status_code=500, detail="no fallback data")

        fallback: List[Quote] = [
            Quote(
                code=i.code,
                name=i.name,
                price=i.price,
                change=i.change,
                change_percent=i.change_percent,
            )
            for i in items
        ]
        db.close()
        set_cache(cache_key, fallback)
        return fallback


# =====================================================================
# 2. 单条实时行情（指数 / 股票）（缓存 + 写 DB + DB 兜底）
# =====================================================================
@app.get("/api/quote/by_code", response_model=Quote)
def get_quote_by_code(
    type: Literal["index", "stock"],
    code: str,
):
    """
    按代码获取一条实时行情（指数 / 股票），带缓存 + DB 兜底。
    股票部分兼容两种返回格式：
      1）宽表：列名直接是 name / current_price 等
      2）纵向 KV 表：只有 item / value，两列，key 在 item 里
    """

    cache_key = f"quote_{type}_{code}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    try:
        # ========================
        # 1) 指数：新浪接口
        # ========================
        if type == "index":
            df = get_index_spot_df()
            if df is None or df.empty:
                raise Exception("failed to load index data from sina")

            symbol = map_index_code_to_symbol(code)  # 000001 -> sh000001
            df_sel = df[df["代码"] == symbol]

            if df_sel.empty:
                raise Exception(f"index {code} not found in sina data")

            row = df_sel.iloc[0]
            pct = row["涨跌幅"]
            if isinstance(pct, str):
                pct = pct.replace("%", "")

            result = Quote(
                code=code,
                name=str(row["名称"]),
                price=float(row["最新价"]),
                change=float(row["涨跌额"]),
                change_percent=float(pct),
            )

            # 写入指数表
            db = SessionLocal()
            try:
                db.merge(IndexQuoteModel(
                    code=code,
                    name=str(row["名称"]),
                    price=float(row["最新价"]),
                    change=float(row["涨跌额"]),
                    change_percent=float(pct),
                    updated_at=datetime.utcnow(),
                ))
                db.commit()
            finally:
                db.close()

            set_cache(cache_key, result)
            return result

        # ========================
        # 2) 股票：雪球单票接口
        # ========================
        elif type == "stock":
            df = get_stock_spot_df()
            if df is None or df.empty:
                raise Exception("failed to load stock spot data")

            row_df = df[df["代码"] == code]
            if row_df.empty:
                raise Exception(f"stock {code} not found in spot data")

            row = row_df.iloc[0]
            pct = row["涨跌幅"]
            if isinstance(pct, str):
                pct = pct.replace("%", "").strip()

            result = Quote(
                code=code,
                name=str(row["名称"]),
                price=float(row["最新价"]),
                change=float(row["涨跌额"]),
                change_percent=float(pct),
            )

            db = SessionLocal()
            try:
                db.merge(StockQuoteModel(
                    code=code,
                    name=result.name,
                    price=result.price,
                    change=result.change,
                    change_percent=result.change_percent,
                    updated_at=datetime.utcnow(),
                ))
                db.commit()
            finally:
                db.close()

            set_cache(cache_key, result)
            return result


        # 正常不会走到这里，但做个防御
        else:
            raise HTTPException(status_code=400, detail="invalid type, must be index or stock")

    except HTTPException:
        # 显式抛的 HTTP 错误直接传给 FastAPI
        raise

    except Exception as e:
        # 统一异常：尝试用数据库兜底
        traceback.print_exc()

        db = SessionLocal()
        try:
            if type == "index":
                item = db.query(IndexQuoteModel).filter_by(code=code).first()
            elif type == "stock":
                item = db.query(StockQuoteModel).filter_by(code=code).first()
            else:
                item = None
        finally:
            db.close()

        if not item:
            # 源头挂了、DB 里也没有，就只能报 500
            raise HTTPException(status_code=500, detail=f"backend error and no fallback: {e}")

        result = Quote(
            code=item.code,
            name=item.name,
            price=item.price,
            change=item.change,
            change_percent=item.change_percent,
        )
        set_cache(cache_key, result)
        return result



# =====================================================================
# 3. K 线（缓存 + 写 DB + DB 兜底）
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
        # ===== 1. 抓取数据 =====
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
            raise Exception("no kline source")

        # ===== 2. 列名统一 =====
        mapping = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
        df = df.rename(columns=mapping)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # ===== 3. 聚合 =====
        if period == "week":
            df = (
                df.resample("W-FRI", on="date")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
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
                    "volume": "sum",
                })
                .dropna()
                .reset_index()
            )

        df = df.tail(limit)

        # ===== 4. 写入数据库 =====
        db = SessionLocal()
        try:
            for _, row in df.iterrows():
                db_item = KlineModel(
                    type=type,
                    code=code,
                    date=row["date"].date(),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
                db.merge(db_item)
            db.commit()
        finally:
            db.close()

        # ===== 5. 返回结果 =====
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

    except Exception:
        traceback.print_exc()

        # ===== 兜底：数据库 =====
        db = SessionLocal()
        try:
            items = (
                db.query(KlineModel)
                .filter_by(type=type, code=code)
                .order_by(KlineModel.date.desc())
                .limit(limit)
                .all()
            )
        finally:
            db.close()

        if not items:
            raise HTTPException(status_code=500, detail="no fallback kline")

        # 注意：这里要反转一次，保证按日期升序返回
        return [
            KlinePoint(
                date=str(i.date),
                open=i.open,
                high=i.high,
                low=i.low,
                close=i.close,
                volume=i.volume,
            )
            for i in reversed(items)
        ]


# =====================================================================
# 入口
# =====================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
