from fastapi import FastAPI, HTTPException
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


# 实时指数接口
@app.get("/api/indices/realtime", response_model=list[StockIndex])
def get_realtime_indices():
    try:
        # 1. 用新的接口：stock_zh_index_spot_em
        #   symbol 参数可选:
        #   "沪深重要指数", "上证系列指数", "深证系列指数", "指数成份", "中证系列指数"
        df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")

        # 调试：打印到后端终端看看
        print("=== ak.stock_zh_index_spot_em columns ===")
        print(df.columns)
        print(df.head())

        # 2. 选择你关心的几个指数代码
        #   注意：这里的代码要和 df["代码"] 里实际显示的一致
        wanted_codes = ["000001", "399001", "399006", "000300", "000688"]
        df_sel = df[df["代码"].isin(wanted_codes)].copy()

        # 如果一个都没匹配上，就先随便拿前 10 个返回，至少前端能看到点东西
        if df_sel.empty:
            df_sel = df.head(10).copy()

        # 3. 转成前端好用的结构
        indices: list[StockIndex] = []
        for _, row in df_sel.iterrows():
            # 涨跌幅可能是带百分号的字符串，先干掉 %
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


if __name__ == "__main__":
    import uvicorn
    # 还是默认 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
