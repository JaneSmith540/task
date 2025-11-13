import pandas as pd
from Data_Handling import DataHandler
from Backtest_Engine import BacktestEngine
from Strategy_Core import MA5Strategy
# 1. 初始化数据处理器（指定数据文件路径）
file_path = r"D:\read\task\机器学习数据.pkl"
data_handler = DataHandler(file_path)

# 2. 初始化回测引擎（传入数据处理器、策略类、初始资金）
backtest_engine = BacktestEngine(
    data_handler=data_handler,
    strategy_class=MA5Strategy,
    initial_cash=100000
)

# 3. 运行回测（指定回测日期范围，需在数据文件的日期范围内）
backtest_engine.run(
    start_date=pd.to_datetime('2018-09-02'),
    end_date=pd.to_datetime('2025-3-15')
)