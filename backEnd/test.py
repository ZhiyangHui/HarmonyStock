import os

for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
          "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)

import akshare as ak
import traceback

print("=== Test: stock_zh_index_spot_sina ===")
print("akshare version:", ak.__version__)

try:
    df = ak.stock_zh_index_spot_sina()
    print("✅ success, shape =", df.shape)
    print(df.head())
except Exception as e:
    print("❌ failed:", repr(e))
    traceback.print_exc()
