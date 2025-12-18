# ===== 0. 在导入业务库之前，先彻底关闭代理环境 =====
import os

# 0.1 清理系统环境变量中可能存在的代理配置
# 防止 requests / urllib 等库自动走代理，导致请求异常或超时
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]:
    os.environ.pop(k, None)

# 0.2 禁用 requests 从环境变量中读取代理配置
# 即使环境变量后来被设置，这里也强制 requests 直连网络
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
# 根据已定义的 ORM 模型，在数据库中创建对应的数据表
# 如果表已存在，不会重复创建
Base.metadata.create_all(bind=engine)


# ===== 通用缓存模块 =====
# 用于缓存一些轻量、通用的中间结果，减少重复计算或请求
CACHE: dict = {}
CACHE_TTL = 30  # 缓存有效期，单位：秒


# ===== 上游“重数据源”缓存（避免重复的昂贵操作） =====
# 这里的“重”主要指：请求慢、数据量大、或对第三方接口压力大的接口

# 指数行情缓存
INDEX_SPOT_CACHE = {
    "data": None,  # 上一次成功获取的指数数据
    "ts": 0,       # 上一次成功获取数据的时间戳
}

# 股票行情缓存
STOCK_SPOT_CACHE = {
    "data": None,      # 上一次成功获取的股票行情数据（通常是 DataFrame）
    "ts": 0.0,         # 上一次成功获取数据的时间戳
    "fail_ts": 0.0,    # 上一次请求失败的时间戳（用于失败降频）
}

# 股票行情失败后的冷却时间
# 如果刚刚失败过，在这段时间内不再频繁重试，避免接口被打爆
STOCK_FAIL_COOLDOWN = 10  # 单位：秒


# ===== 不同数据源的缓存时长 =====
# 指数接口相对轻量，更新频率低一些即可
INDEX_SPOT_TTL = 30   # 指数行情缓存 30 秒

# 股票接口通常更重（数据量大、处理复杂），缓存时间设置更长
STOCK_SPOT_TTL = 50   # 股票行情缓存 50 秒


# ===== 股票代码到行数据的映射缓存 =====
# 用于快速从股票代码定位到对应的数据行，减少反复遍历 DataFrame
STOCK_ROW_MAP = {
    "map": None,   # 结构通常为 dict[code -> 行数据]
    "ts": 0.0,     # 映射缓存的更新时间
}



def get_index_spot_df():
    # 当前时间戳，用于判断缓存是否过期
    now = time.time()

    # 如果缓存里已有数据，且还在有效期内，直接返回缓存
    if INDEX_SPOT_CACHE["data"] is not None and now - INDEX_SPOT_CACHE["ts"] < INDEX_SPOT_TTL:
        return INDEX_SPOT_CACHE["data"]

    # 缓存失效或不存在，重新从上游接口拉取指数行情
    df = ak.stock_zh_index_spot_sina()

    # 上游接口异常或返回空数据，直接抛错，交给上层处理
    if df is None or df.empty:
        raise Exception("index spot source error")

    # 更新缓存数据和时间戳
    INDEX_SPOT_CACHE["data"] = df
    INDEX_SPOT_CACHE["ts"] = now

    # DataFrame缩写
    return df

def get_stock_spot_df():
    now = time.time()

    # 命中有效缓存：直接返回
    if STOCK_SPOT_CACHE["data"] is not None and now - STOCK_SPOT_CACHE["ts"] < STOCK_SPOT_TTL:
        return STOCK_SPOT_CACHE["data"]

    # 如果刚失败过，且手里有旧数据：直接返回旧数据（避免每次请求都阻塞在上游）
    if (STOCK_SPOT_CACHE["data"] is not None and
        now - STOCK_SPOT_CACHE["fail_ts"] < STOCK_FAIL_COOLDOWN):
        return STOCK_SPOT_CACHE["data"]

    sina_err = None

    # 尝试刷新（新浪）
    try:
        df = ak.stock_zh_a_spot()
        if df is not None and not df.empty:
            STOCK_SPOT_CACHE["data"] = df
            STOCK_SPOT_CACHE["ts"] = now
            return df
    except Exception as e:
        sina_err = str(e)

    # 尝试刷新（东财）
    try:
        df2 = ak.stock_zh_a_spot_em()
        if df2 is not None and not df2.empty:
            STOCK_SPOT_CACHE["data"] = df2
            STOCK_SPOT_CACHE["ts"] = now
            return df2
        raise Exception("empty df from em")
    except Exception as e2:
        STOCK_SPOT_CACHE["fail_ts"] = now

        # 刷新失败：如果有旧数据，返回旧数据（核心）
        if STOCK_SPOT_CACHE["data"] is not None:
            return STOCK_SPOT_CACHE["data"]

        # 连旧数据都没有：只能抛错
        raise Exception(f"stock spot upstream failed. sina={sina_err}; em={e2}")


# 根据最新股票行情 DataFrame 构建 code -> 行数据(dict) 的快速索引映射，用于高效按代码查行情
def get_stock_row_map() -> dict:
    # 获取带缓存与降级保护的股票实时行情 DataFrame
    df = get_stock_spot_df()
    now = time.time()

    # 如果当前 row_map 已构建且对应的行情时间戳未变化，直接复用（避免重复遍历 DataFrame）
    if STOCK_ROW_MAP["map"] is not None and STOCK_ROW_MAP["ts"] == STOCK_SPOT_CACHE["ts"]:
        return STOCK_ROW_MAP["map"]

    # 防御性校验：行情表必须包含“代码”列，否则说明上游结构异常
    if "代码" not in df.columns:
        raise Exception(f"spot df missing '代码' column, got: {list(df.columns)}")

    # 提取股票代码列并统一为字符串，截取末尾 6 位作为标准股票代码
    code_series = df["代码"].astype(str).str.strip().str[-6:]

    # 构建 code -> 行数据(dict) 的映射，后续按代码查询可 O(1) 访问
    row_map = {}
    for i in range(len(df)):
        code6 = code_series.iat[i]          # 当前行对应的 6 位股票代码
        row_map[code6] = df.iloc[i].to_dict()  # 将整行转为 dict，便于后续直接取字段

    # 缓存最新构建的 row_map，并记录对应的行情时间戳
    STOCK_ROW_MAP["map"] = row_map
    STOCK_ROW_MAP["ts"] = STOCK_SPOT_CACHE["ts"]

    # 返回股票代码到行数据的映射表
    return row_map



# 从本地内存缓存中读取指定 key 的数据，若未过期则返回，否则返回 None
def get_cache(key: str):
    # 从全局 CACHE 字典中取出缓存项
    item = CACHE.get(key)

    # 如果缓存存在且未超过 TTL（生存时间），直接返回缓存数据
    if item and time.time() - item["ts"] < CACHE_TTL:
        return item["data"]

    # 缓存不存在或已过期
    return None


# 将数据写入本地内存缓存，并记录当前时间戳用于 TTL 判断
def set_cache(key: str, data):
    # 保存数据和写入时间，用于后续缓存命中判断
    CACHE[key] = {"data": data, "ts": time.time()}

# ===== FastAPI 初始化 =====
app = FastAPI(title="HarmonyStock Backend")

# 添加 CORS 中间件，允许前端 / ArkUI / 桌面卡片跨域访问接口
app.add_middleware(
    CORSMiddleware,
    # 允许所有来源访问（桌面卡片、App、WebView 都需要）
    allow_origins=["*"],
    # 允许携带凭证信息（为后续鉴权、登录预留）
    allow_credentials=True,
    # 允许所有 HTTP 方法（GET / POST / PUT / DELETE 等）
    allow_methods=["*"],
    # 允许所有请求头，避免前端被浏览器拦截
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "HarmonyStock backend is running"}


# FastAPI 启动事件：在服务启动完成前执行一次初始化逻辑
@app.on_event("startup")
def warm_up_cache() -> None:
    # 预热指数行情缓存，避免第一次请求时阻塞在上游接口
    try:
        get_index_spot_df()
        print("[startup] index spot warmed")
    except Exception as e:
        # 指数预热失败不影响服务启动，只记录日志
        print("[startup] index warm failed:", e)

    # 预热股票行情缓存，并构建 code -> 行数据映射表
    try:
        get_stock_spot_df()      # 拉取股票实时行情并写入缓存
        get_stock_row_map()      # 基于行情 df 构建快速查询用的 row_map
        print("[startup] stock spot warmed + row map built")
    except Exception as e:
        # 股票预热失败同样不阻断启动，允许后续请求再触发刷新
        print("[startup] stock warm failed:", e)


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
# 将 6 位 A 股股票代码映射为行情接口常用的 symbol（sh / sz 前缀）
def map_stock_code_to_symbol(code: str) -> str:
    # 以上交所股票（6/9 开头）使用 sh，其余默认深交所 sz
    return f"sh{code}" if code[0] in ("6", "9") else f"sz{code}"


# 将指数代码映射为行情接口常用的 symbol（根据指数所属交易所判断）
def map_index_code_to_symbol(code: str) -> str:
    # 深证指数统一以 399 开头，其余指数视为上证指数
    return f"sz{code}" if code.startswith("399") else f"sh{code}"


# 将股票代码映射为雪球（Xueqiu）接口使用的 symbol（大写 SH / SZ）
def map_stock_code_to_xq_symbol(code: str) -> str:
    # 深交所股票：0 / 2 / 3 开头
    if code[0] in ("0", "2", "3"):
        return f"SZ{code}"
    # 上交所股票：6 / 9 开头
    if code[0] in ("6", "9"):
        return f"SH{code}"
    # 非法或不支持的股票代码直接抛错，避免静默失败
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



# GET /api/quote：返回指数或股票的“实时行情列表”，支持可选 codes=逗号分隔代码；带内存缓存 + DB 落库 + 异常时 DB 兜底
@app.get("/api/quote", response_model=List[Quote])
def get_quote(
    type: str = Query("index"),          # 查询参数：index / stock（默认 index）
    codes: Optional[str] = None          # 查询参数：可选，"300750,601012" 这种逗号分隔
):
    # 说明：实时行情列表（type=index：默认 3 个指数；type=stock：默认 3 只股票；都支持前端传 codes 覆盖）
    cache_key = f"quote_list_{type}_{codes}"   # 缓存 key：把 type + codes 组合起来区分不同请求
    cached = get_cache(cache_key)              # 先查内存缓存（TTL 在 get_cache 里控制）
    if cached:
        return cached                          # 命中缓存直接返回，避免频繁请求上游/写DB

    db = SessionLocal()                        # 创建一次 DB 会话（本请求内复用）
    try:
        # 1) 指数行情：从指数 spot DF 里筛选需要的指数代码
        if type == "index":
            df = get_index_spot_df()           # 拉取（或命中缓存）指数全量行情 DataFrame
            if df is None or df.empty:
                raise Exception("index source error")

            # codes 参数优先：传了就按传入的列表；没传就用默认 3 个指数
            if codes:
                codes_list = [c.strip() for c in codes.split(",") if c.strip()]
            else:
                codes_list = ["000001", "399001", "399006"]  # 默认：上证/深成/创业板（按你的定义）

            symbols = [map_index_code_to_symbol(c) for c in codes_list]  # 转成 DF 中使用的带前缀 symbol（sh/sz）
            df_sel = df[df["代码"].isin(symbols)]                         # 在 DF 中按“代码”列过滤出目标行

            if df_sel.empty:
                raise Exception("no index rows after filter")

            result: List[Quote] = []           # 最终返回给前端的 Quote 列表

            for _, row in df_sel.iterrows():   # 遍历每一行指数数据
                pct = row["涨跌幅"]            # 涨跌幅可能是 "1.23%" 或数值
                if isinstance(pct, str):
                    pct = pct.replace("%", "") # 去掉百分号，转成可 float 的字符串

                q = Quote(
                    code=row["代码"][-6:],     # DF 的代码可能含前缀（如 sh000001），这里只取后 6 位
                    name=row["名称"],
                    price=float(row["最新价"]),
                    change=float(row["涨跌额"]),
                    change_percent=float(pct),
                )
                result.append(q)               # 加入返回列表

                # 写入/更新数据库：IndexQuoteModel 作为落库兜底数据源
                db_item = IndexQuoteModel(
                    code=q.code,
                    name=q.name,
                    price=q.price,
                    change=q.change,
                    change_percent=q.change_percent,
                    updated_at=datetime.utcnow(),  # 记录更新时间（UTC）
                )
                db.merge(db_item)              # merge：存在则更新，不存在则插入（按主键/唯一键规则）

            db.commit()                        # 提交本次指数批量写入
            set_cache(cache_key, result)       # 写入内存缓存，后续相同请求直接返回
            return result

        # 2) 股票行情：从股票 spot DF 里筛选需要的股票代码
        elif type == "stock":
            # codes 参数优先：传了就按传入列表；没传就用默认 3 只股票
            if codes:
                stock_codes = [c.strip() for c in codes.split(",") if c.strip()]
            else:
                stock_codes = ["300750", "601012", "688981"]  # 默认：宁德时代/隆基/中芯（按你的预设）

            df = get_stock_spot_df()           # 拉取（或命中缓存）股票全量行情 DataFrame（重接口，所以你做了更强缓存）
            if df is None or df.empty:
                raise Exception("stock spot source error")

            # 校验 DF 是否包含你依赖的列名，避免源切换导致 KeyError
            needed = ["代码", "名称", "最新价", "涨跌额", "涨跌幅"]
            for col in needed:
                if col not in df.columns:
                    raise Exception(f"unexpected stock spot columns, missing: {col}, got: {list(df.columns)}")

            df_sel = df[df["代码"].isin(stock_codes)]  # 过滤出目标股票行
            if df_sel.empty:
                raise Exception("no stock rows after filter")

            result: List[Quote] = []           # 最终返回给前端的 Quote 列表
            for _, row in df_sel.iterrows():
                pct = row["涨跌幅"]            # 同样处理涨跌幅字段
                if isinstance(pct, str):
                    pct = pct.replace("%", "").strip()

                q = Quote(
                    code=str(row["代码"]),     # 股票代码一般就是 6 位字符串
                    name=str(row["名称"]),
                    price=float(row["最新价"]),
                    change=float(row["涨跌额"]),
                    change_percent=float(pct),
                )
                result.append(q)

                # 写入/更新数据库：StockQuoteModel 作为落库兜底数据源
                db.merge(StockQuoteModel(
                    code=q.code,
                    name=q.name,
                    price=q.price,
                    change=q.change,
                    change_percent=q.change_percent,
                    updated_at=datetime.utcnow(),
                ))

            db.commit()                        # 提交本次股票批量写入
            set_cache(cache_key, result)       # 写入内存缓存
            return result

        # 3) 非法 type：直接返回 400
        else:
            raise HTTPException(status_code=400, detail="only index or stock supported")

    except Exception:
        traceback.print_exc()                  # 打印异常堆栈，便于你在服务端定位

        # 异常兜底：上游失败时，从数据库读取最近一次写入的行情作为 fallback
        if type == "index":
            items = db.query(IndexQuoteModel).all()
        elif type == "stock":
            items = db.query(StockQuoteModel).all()
        else:
            items = []

        if not items:
            db.close()
            raise HTTPException(status_code=500, detail="no fallback data")  # 连 DB 都没数据：只能报 500

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
        db.close()                             # 兜底路径显式关闭连接（正常路径建议也加 finally 统一关闭）
        set_cache(cache_key, fallback)         # 兜底结果也写缓存，避免短时间内反复打 DB
        return fallback




# /api/kline：返回指定标的的 K 线（先查内存缓存；再抓上游并落库；失败时用 DB 兜底）
# - type: index/stock
# - code: 6 位代码
# - period: day/week/month（周/月通过 resample 聚合）
# - limit: 返回最近 N 根
@app.get("/api/kline", response_model=List[KlinePoint])
def get_kline(
    type: Literal["index", "stock"],                      # 查询参数：指数 or 股票
    code: str,                                            # 查询参数：6 位代码
    period: Literal["day", "week", "month"] = "day",      # 查询参数：K 线周期（默认日K）
    limit: int = 60,                                      # 查询参数：返回最近多少根（默认 60）
):
    cache_key = f"kline_{type}_{code}_{period}_{limit}"   # 内存缓存 key：同参数请求命中同一份缓存
    cached = get_cache(cache_key)                          # 先查内存缓存，避免重复抓取/聚合
    if cached:
        return cached

    try:
        # 1）抓取上游原始日线数据：指数/股票用不同 AkShare 接口
        if type == "index":
            symbol = map_index_code_to_symbol(code)        # 指数代码转 AkShare 需要的 symbol（sh/sz 前缀）
            df = ak.stock_zh_index_daily(symbol=symbol)    # 获取指数日线 DF（含 日期/开高低收/量 等列）
        else:
            symbol = map_stock_code_to_symbol(code)        # 股票代码转 AkShare 需要的 symbol（sh/sz 前缀）
            end = datetime.today()                         # 结束日期：今天
            start = end - timedelta(days=365 * 5)          # 起始日期：往前 5 年（给周/月聚合留足样本）
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start.strftime("%Y%m%d"),       # AkShare 期望 YYYYMMDD
                end_date=end.strftime("%Y%m%d"),
            )

        if df is None or df.empty:
            raise Exception("no kline source")             # 上游没数据：走 except，进入 DB 兜底

        # 2）统一列名：把中文列统一成 date/open/high/low/close/volume，便于后续聚合与落库
        mapping = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
        df = df.rename(columns=mapping)

        df["date"] = pd.to_datetime(df["date"])            # 确保 date 列是 datetime，才能 resample
        df = df.sort_values("date")                        # 确保按时间升序（first/last 聚合才正确）

        # 3）按周期聚合：日K不聚合；周K/月K用 resample 做 OHLCV 聚合
        if period == "week":
            df = (
                df.resample("W-FRI", on="date")            # 周频，以周五为周期结束点（更贴近 A 股交易周）
                .agg({
                    "open": "first",                       # 周开盘：该周期第一天开盘
                    "high": "max",                         # 周最高：周期内最高
                    "low": "min",                          # 周最低：周期内最低
                    "close": "last",                       # 周收盘：周期最后一天收盘
                    "volume": "sum",                       # 周成交量：周期内求和
                })
                .dropna()                                   # 去掉不完整周期（例如最前面不足一周）
                .reset_index()                               # resample 后 index 是 date，这里还原成列
            )
        elif period == "month":
            df = (
                df.resample("M", on="date")                # 月频（自然月）
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

        df = df.tail(limit)                                 # 只取最近 limit 根，减少返回体积与 DB 写入量

        # 4）落库：把本次算出来的 K 线写入 KlineModel，供后续兜底与复用
        db = SessionLocal()
        try:
            for _, row in df.iterrows():
                db_item = KlineModel(
                    type=type,                              # index/stock
                    code=code,                              # 6 位代码
                    date=row["date"].date(),                # 只存日期部分（date 类型）
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
                db.merge(db_item)                           # upsert：已有则更新，没有则插入
            db.commit()
        finally:
            db.close()

        # 5）组装响应：把 DF 行转成 API 输出的 KlinePoint 列表（日期升序）
        result = [
            KlinePoint(
                date=str(row["date"].date()),               # 返回字符串日期，前端易处理
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for _, row in df.iterrows()
        ]

        set_cache(cache_key, result)                        # 写入内存缓存，短期内重复请求直接命中
        return result

    except Exception:
        traceback.print_exc()

        # 上游失败/聚合失败：DB 兜底（取最近 limit 条）
        db = SessionLocal()
        try:
            items = (
                db.query(KlineModel)
                .filter_by(type=type, code=code)            # 只取该标的、该类型的记录
                .order_by(KlineModel.date.desc())           # 先按日期倒序取最新的 limit 条
                .limit(limit)
                .all()
            )
        finally:
            db.close()

        if not items:
            raise HTTPException(status_code=500, detail="no fallback kline")  # DB 也没数据：直接报错

        # DB 查询是倒序（新->旧），但前端画图一般需要升序（旧->新），所以这里 reversed 一次
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



# 入口
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
