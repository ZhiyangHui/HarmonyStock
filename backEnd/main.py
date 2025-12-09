from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import akshare as ak
import pandas as pd
import traceback
from datetime import datetime, timedelta
from typing import Literal


app = FastAPI(title="HarmonyStock Backend")

# CORS，给鸿蒙前端用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 开发阶段全部放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 根路由：健康检查
@app.get("/")
def root():
    return {"message": "HarmonyStock backend is running"}


# ===== 指数结构 =====
class StockIndex(BaseModel):
    code: str
    name: str
    price: float
    change: float
    change_percent: float


# ===== K 线点结构 =====
class KlinePoint(BaseModel):
    date: str      # 日期，比如 "2025-12-08"
    open: float
    high: float
    low: float
    close: float
    volume: float


# ===== 实时指数接口（你之前已经在用的） =====
@app.get("/api/indices/realtime", response_model=list[StockIndex])
def get_realtime_indices():
    try:
        # 用东方财富的指数现货接口
        df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")

        print("=== stock_zh_index_spot_em columns ===")
        print(df.columns)
        print(df.head())

        # 你关心的几个指数代码（和前端一致）
        wanted_codes = ["000001", "399001", "399006", "000300", "000688"]
        df_sel = df[df["代码"].isin(wanted_codes)].copy()

        # 如果一个都没匹配上，就先返回前 10 个
        if df_sel.empty:
            df_sel = df.head(10).copy()

        indices: list[StockIndex] = []
        for _, row in df_sel.iterrows():
            raw_pct = row["涨跌幅"]
            if isinstance(raw_pct, str):
                raw_pct = raw_pct.replace("%", "")
            change_percent = float(raw_pct)

            idx = StockIndex(
                code=str(row["代码"]),
                name=str(row["名称"]),
                price=float(row["最新价"]),
                change=float(row["涨跌额"]),
                change_percent=change_percent,
            )
            indices.append(idx)

        return indices

    except Exception as e:
        print("ERROR in /api/indices/realtime:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"backend error: {e}")


# ===== 按代码查一个指数（你已经在前端用来“搜索指数”了） =====
@app.get("/api/index/by_code", response_model=StockIndex)
def get_index_by_code(code: str = Query(..., min_length=6, max_length=6)):
    try:
        df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
        row = df[df["代码"] == code].iloc[0]
    except IndexError:
        raise HTTPException(status_code=404, detail="index not found")
    except Exception as e:
        print("ERROR in /api/index/by_code:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"backend error: {e}")

    raw_pct = row["涨跌幅"]
    if isinstance(raw_pct, str):
        raw_pct = raw_pct.replace("%", "")
    change_percent = float(raw_pct)

    return StockIndex(
        code=str(row["代码"]),
        name=str(row["名称"]),
        price=float(row["最新价"]),
        change=float(row["涨跌额"]),
        change_percent=change_percent,
    )


# ===== 新增：指数 K 线接口 =====
# 返回：日K / 周K / 月K 的 K 线数组

@app.get("/api/indices/kline", response_model=list[KlinePoint])
def get_index_kline(
    code: str = Query(..., min_length=6, max_length=6),
    period: Literal["day", "week", "month"] = "day",
    limit: int = 60,
):
    """
    指数日 K 线数据
    - code: 指数代码，例如 "000001"（上证）、"399001"（深证）、"399006"（创业板）等
    - period: 暂时只实现 "day"，传 week/month 目前也按 day 处理
    - limit: 取最近多少根 K 线
    """
    try:
        # 1. 把纯数字 code 映射成 akshare 需要的 symbol（带市场前缀）
        #   上证/沪市：一般用 "sh" 开头
        #   深证：一般用 "sz" 开头
        if code.startswith("399"):
            # 深市指数（例如 399001、399006）
            symbol = f"sz{code}"
        else:
            # 其他常见指数先按沪市处理（000001, 000300, 000688 等）
            symbol = f"sh{code}"

        # 2. 拉取全部日 K 线数据（接口只支持 symbol 这一个参数）
        df = ak.stock_zh_index_daily(symbol=symbol)
        # df 列：["date", "open", "high", "low", "close", "volume"]

        # 3. 按日期升序排一下，保证时间顺序正确
        df = df.sort_values("date")

        # 4. 只取最后 limit 条
        if limit > 0:
            df = df.tail(limit)

        # 5. 转成前端要的结构
        points: list[KlinePoint] = []
        for _, row in df.iterrows():
            p = KlinePoint(
                date=str(row["date"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            points.append(p)

        return points

    except Exception as e:
        print("ERROR in /api/indices/kline:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"backend error: {e}")



if __name__ == "__main__":
    import uvicorn
    # 本地开发直接跑
    uvicorn.run(app, host="0.0.0.0", port=8000)
