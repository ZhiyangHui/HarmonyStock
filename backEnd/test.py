# test_quote_minimal.py
import os

# 尽可能清掉 Python 进程里能看到的代理变量
for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]:
    os.environ.pop(k, None)

import akshare as ak
import traceback

print("=== akshare quote minimal test ===")
print("akshare version:", ak.__version__)

try:
    print(">>> 调用 stock_zh_index_spot_em ...")
    df = ak.stock_zh_index_spot_em(symbol="沪深重要指数")
    print("✅ success, shape =", df.shape)
    print(df.head())
except Exception as e:
    print("❌ failed:", repr(e))
    traceback.print_exc()
