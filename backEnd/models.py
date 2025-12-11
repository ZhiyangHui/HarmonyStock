# models.py
from sqlalchemy import Column, String, Float, DateTime, Integer, Date
from datetime import datetime
from db import Base


# ================================
# 1. 指数实时行情表
# ================================
class IndexQuoteModel(Base):
    __tablename__ = "index_quote"

    code = Column(String, primary_key=True)  # 000001
    name = Column(String)
    price = Column(Float)
    change = Column(Float)
    change_percent = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


# ================================
# 2. 股票实时行情表
# ================================
class StockQuoteModel(Base):
    __tablename__ = "stock_quote"

    code = Column(String, primary_key=True)  # 300750
    name = Column(String)
    price = Column(Float)
    change = Column(Float)
    change_percent = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)


# ================================
# 3. K 线表（指数 & 股票共用）
# ================================
class KlineModel(Base):
    __tablename__ = "kline"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String)   # index / stock
    code = Column(String)   # 000001 or 300750
    date = Column(Date)     # yyyy-mm-dd

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
