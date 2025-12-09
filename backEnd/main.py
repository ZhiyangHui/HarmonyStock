from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import akshare as ak
import traceback

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


# 指数结构
class StockIndex(BaseModel):
    code: str
    name: str
    price: float
    change: float
    change_percent: float


# ========= 1. 实时指数列表 =========
@app.get("/api/indices/realtime", response_model=list[StockIndex])
def get_realtime_indices():
    try:
        # 用新的接口：沪深重要指数
        df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")

        print("=== /api/indices/realtime columns ===")
        print(df.columns)
        print(df.head())

        # 关心的几个指数
        wanted_codes = ["000001", "399001", "399006", "000300", "000688"]
        df_sel = df[df["代码"].isin(wanted_codes)].copy()

        if df_sel.empty:
            # 万一匹配不到，就返回前 10 个，保证前端能看到东西
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


# ========= 2. 根据代码查询单个指数 =========
@app.get("/api/index/by_code", response_model=StockIndex)
def get_index_by_code(
    code: str = Query(..., min_length=6, max_length=6, description="指数代码，如 000001"),
):
    """
    通过 ?code=000001 这种方式查询单个指数
    """
    try:
        # 跟上面保持一致，用 em 接口
        df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")

        print("=== /api/index/by_code, code =", code, "===")

        # 从表里找这一行
        match_df = df[df["代码"] == code]

        if match_df.empty:
            # 没找到
            raise HTTPException(status_code=404, detail="index not found")

        row = match_df.iloc[0]

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

    except HTTPException:
        # 直接往外抛自定义 404
        raise
    except Exception as e:
        print("ERROR in /api/index/by_code:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"backend error: {e}")


if __name__ == "__main__":
    import uvicorn
    # 还是默认 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
