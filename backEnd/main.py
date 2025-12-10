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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import akshare as ak
import pandas as pd
import traceback
from datetime import datetime, timedelta
from typing import Literal, Optional, List


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


# ===== 通用行情结构（指数 / 股票一样）=====
class Quote(BaseModel):
    code: str
    name: str
    price: float
    change: float
    change_percent: float


# 为了兼容前端已有类型，再定义两个“别名模型”
class StockIndex(Quote):
    pass


class StockQuote(Quote):
    pass


# ===== K 线点结构 =====
class KlinePoint(BaseModel):
    date: str      # 日期，比如 "2025-12-08"
    open: float
    high: float
    low: float
    close: float
    volume: float


# ===== 辅助函数：个股代码 -> 东财 symbol（带市场前缀） =====
def map_stock_code_to_symbol(code: str) -> str:
    """
    A 股个股代码 -> 东财 symbol
    - 6/9 开头 -> 上交所 -> sh
    - 其他默认深交所 -> sz
    """
    c = code.strip()
    if len(c) != 6:
        raise ValueError(f"invalid stock code: {code}")

    if c[0] in ("6", "9"):
        return f"sh{c}"
    else:
        return f"sz{c}"


# =====================================================================
# 一、通用实时行情接口：指数 / 股票 共用
# =====================================================================

@app.get("/api/quote", response_model=List[Quote])
def get_quote(
    type: Literal["index", "stock"] = Query(..., description="index 或 stock"),
    codes: Optional[str] = Query(
        default=None,
        description="可选，逗号分隔的代码列表，例如 '000001,399001,300750'"
    )
):
    """
    通用实时行情：
    - type = index：指数
    - type = stock：个股
    - codes：可选；不传则按默认逻辑返回一批
    """
    try:
        if type == "index":
            # 指数：东方财富 沪深重要指数
            df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")

            if codes:
                wanted = [c.strip() for c in codes.split(",") if c.strip()]
                df_sel = df[df["代码"].isin(wanted)].copy()
            else:
                # 没传 codes，就按你之前的“关注几个主流指数”为主
                wanted_codes = ["000001", "399001", "399006", "000300", "000688"]
                df_sel = df[df["代码"].isin(wanted_codes)].copy()
                if df_sel.empty:
                    df_sel = df.head(10).copy()

        elif type == "stock":
            # 个股：A 股实时行情
            df = ak.stock_zh_a_spot_em()

            if codes:
                wanted = [c.strip() for c in codes.split(",") if c.strip()]
                df_sel = df[df["代码"].isin(wanted)].copy()
            else:
                # 不指定，则给前 50 只作为示例
                df_sel = df.head(50).copy()
        else:
            raise HTTPException(status_code=400, detail="invalid type, must be index or stock")

        if df_sel.empty:
            raise HTTPException(status_code=404, detail="no quote data")

        quotes: List[Quote] = []
        for _, row in df_sel.iterrows():
            raw_pct = row["涨跌幅"]
            if isinstance(raw_pct, str):
                raw_pct = raw_pct.replace("%", "")
            change_percent = float(raw_pct)

            q = Quote(
                code=str(row["代码"]),
                name=str(row["名称"]),
                price=float(row["最新价"]),
                change=float(row["涨跌额"]),
                change_percent=change_percent,
            )
            quotes.append(q)

        return quotes

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in /api/quote:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"backend error: {e}")


@app.get("/api/quote/by_code", response_model=Quote)
def get_quote_by_code(
    type: Literal["index", "stock"] = Query(..., description="index 或 stock"),
    code: str = Query(..., min_length=6, max_length=6),
):
    """
    通用：按代码查一条实时行情（指数 / 股票）
    """
    try:
        if type == "index":
            df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
        elif type == "stock":
            df = ak.stock_zh_a_spot_em()
        else:
            raise HTTPException(status_code=400, detail="invalid type, must be index or stock")

        df_sel = df[df["代码"] == code]
        if df_sel.empty:
            raise HTTPException(status_code=404, detail="quote not found")

        row = df_sel.iloc[0]
        raw_pct = row["涨跌幅"]
        if isinstance(raw_pct, str):
            raw_pct = raw_pct.replace("%", "")
        change_percent = float(raw_pct)

        return Quote(
            code=str(row["代码"]),
            name=str(row["名称"]),
            price=float(row["最新价"]),
            change=float(row["涨跌额"]),
            change_percent=change_percent,
        )

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in /api/quote/by_code:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"backend error: {e}")


# =====================================================================
# 二、通用 K 线接口：指数 / 股票 共用
# =====================================================================

@app.get("/api/kline", response_model=List[KlinePoint])
def get_kline(
    type: Literal["index", "stock"] = Query(..., description="index 或 stock"),
    code: str = Query(..., min_length=6, max_length=6),
    period: Literal["day", "week", "month"] = "day",
    limit: int = 60,
):
    """
    通用 K 线数据：
    - type: "index" 指数，"stock" 个股
    - code: 代码，例如 "000001"（上证）或 "300750"（宁德时代）
    - period: "day" 日K、"week" 周K、"month" 月K
    - limit: 取最近多少根 K 线
    """
    try:
        # 1. 先根据 type + code 选出合适的原始日线 DataFrame
        if type == "index":
            # 指数：使用 stock_zh_index_daily
            if code.startswith("399"):
                symbol = f"sz{code}"   # 深市指数
            else:
                symbol = f"sh{code}"   # 其他默认沪市

            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail="no kline data for index")

        elif type == "stock":
            # 个股：使用 stock_zh_a_daily，需要 symbol 带市场前缀
            symbol = map_stock_code_to_symbol(code)
            end = datetime.today()
            start = end - timedelta(days=365 * 5)
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail="no kline data for stock")

        else:
            raise HTTPException(status_code=400, detail="invalid type, must be index or stock")

        # 2. 统一列名
        date_col = "date" if "date" in df.columns else "日期"
        open_col = "open" if "open" in df.columns else "开盘"
        high_col = "high" if "high" in df.columns else "最高"
        low_col = "low" if "low" in df.columns else "最低"
        close_col = "close" if "close" in df.columns else "收盘"
        volume_col = "volume" if "volume" in df.columns else "成交量"

        df = df.rename(
            columns={
                date_col: "date",
                open_col: "open",
                high_col: "high",
                low_col: "low",
                close_col: "close",
                volume_col: "volume",
            }
        )

        # 3. 日期转 datetime，按日期排序
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        if df.empty:
            raise HTTPException(status_code=404, detail="no kline data after sort")

        # 4. 按 period 聚合
        if period == "day":
            df_used = df
        elif period == "week":
            # 以周为单位重采样，周五为一周结束
            df_used = (
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
            # 以自然月重采样
            df_used = (
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
        else:
            df_used = df

        # 5. 再按日期升序一次（保险），只要最后 limit 根
        df_used = df_used.sort_values("date")
        if limit > 0:
            df_used = df_used.tail(limit)

        # 6. 转成前端要的结构
        points: List[KlinePoint] = []
        for _, row in df_used.iterrows():
            p = KlinePoint(
                date=str(row["date"].date()),   # "YYYY-MM-DD"
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            points.append(p)

        return points

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in /api/kline:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"backend error: {e}")


if __name__ == "__main__":
    import uvicorn
    # 本地开发直接跑
    uvicorn.run(app, host="0.0.0.0", port=8000)
