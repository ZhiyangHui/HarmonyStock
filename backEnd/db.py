# db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite 数据库文件路径
# 会在项目根目录生成 stock.db 文件
DATABASE_URL = "sqlite:///./stock.db"

# 创建数据库引擎
# check_same_thread=False：允许多线程访问，是 FastAPI + SQLite 必须的设置
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # 若想调试 SQL，可改成 True
)

# 创建 SessionLocal，用于每个请求访问数据库
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# 所有 ORM 模型继承自这个 Base
Base = declarative_base()
