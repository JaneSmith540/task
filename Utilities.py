import pandas as pd


class Log:
    @staticmethod
    def info(msg):
        print(f"[INFO] {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


log = Log()  # 实例化log，供策略调用

# 在关键位置添加时间戳记录
import time
import Data_Handling

def monitor_function_calls():
    # 监控 get_price 调用频率和时间
    original_get_price = Data_Handling.get_price

    def monitored_get_price(*args, **kwargs):
        start = time.time()
        result = original_get_price(*args, **kwargs)
        end = time.time()
        print(f"get_price 调用 - 参数: {args}, 执行时间: {end - start:.4f} 秒")
        return result

    Data_Handling.get_price = monitored_get_price
