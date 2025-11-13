import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Performance_Analysis import PerformanceAnalysis


class BacktestVisualization:
    def __init__(self, account, strategy_returns=None):
        self.account = account
        self.strategy_returns = strategy_returns

    def calculate_returns(self):
        """计算策略收益率"""
        if self.strategy_returns is None:
            self.strategy_returns = pd.Series(
                self.account.total_assets, index=self.account.dates
            ).pct_change().fillna(0)
        return self.strategy_returns

    def load_benchmark_data(self):
        """加载中证500指数数据"""
        try:
            # 读取中证500指数数据
            benchmark_file = r"D:\read\task\中证500指数_201801-202506.csv"
            benchmark_data = pd.read_csv(benchmark_file)

            # 转换日期格式
            benchmark_data['trade_date'] = pd.to_datetime(benchmark_data['trade_date'], format='%Y%m%d')

            # 设置日期为索引
            benchmark_data = benchmark_data.set_index('trade_date')

            # 按日期排序
            benchmark_data = benchmark_data.sort_index()

            return benchmark_data
        except Exception as e:
            print(f"加载基准数据失败: {e}")
            return None

    def calculate_benchmark_returns(self, start_date, end_date):
        """计算中证500指数在指定时间范围内的收益率"""
        benchmark_data = self.load_benchmark_data()

        if benchmark_data is None:
            return None

        # 筛选指定时间范围的数据
        benchmark_data = benchmark_data[(benchmark_data.index >= start_date) & (benchmark_data.index <= end_date)]

        if len(benchmark_data) == 0:
            return None

        # 计算收益率
        benchmark_returns = benchmark_data['close'].pct_change().fillna(0)

        return benchmark_returns

    def plot_results(self):
        """绘制回测结果 - 修复版本"""
        # 创建绩效分析对象
        performance = PerformanceAnalysis(self.account)

        # 使用PerformanceAnalysis类的方法获取收益率数据
        strategy_returns = performance.strategy_returns

        # 计算每日累计收益率
        strategy_cumulative = performance.cumulative_returns

        # 调试信息
        print("=== 调试信息 ===")
        print(f"总资产序列长度: {len(self.account.total_assets)}")
        print(f"日期序列长度: {len(self.account.dates)}")
        print(f"收益率序列长度: {len(strategy_returns)}")
        print(f"累计收益率序列长度: {len(strategy_cumulative)}")
        print(f"总资产范围: {min(self.account.total_assets):.2f} ~ {max(self.account.total_assets):.2f}")
        print(f"收益率范围: {strategy_returns.min():.4f} ~ {strategy_returns.max():.4f}")
        print(f"累计收益率范围: {strategy_cumulative.min():.4f} ~ {strategy_cumulative.max():.4f}")

        # 获取回测时间范围
        start_date = self.account.dates[0]
        end_date = self.account.dates[-1]

        # 计算中证500指数收益率
        benchmark_returns = self.calculate_benchmark_returns(start_date, end_date)

        plt.figure(figsize=(12, 6))

        # 绘制累计收益率曲线
        plt.plot(self.account.dates, strategy_cumulative, label='Strategy Cumulative Return', linewidth=2)

        # 如果有基准数据，也绘制基准收益率曲线
        if benchmark_returns is not None:
            # 确保日期对齐
            common_dates = strategy_cumulative.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                benchmark_cumulative = (1 + benchmark_returns.loc[common_dates]).cumprod() - 1
                plt.plot(common_dates, benchmark_cumulative, label='CSI 500 Index Cumulative Return', linewidth=2)

        # 在图表标题中显示总收益率
        total_return = performance.get_total_return()
        plt.title(f'Strategy vs CSI 500 Index Cumulative Returns\nTotal Return: {total_return:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_performance(self):
        """打印绩效指标"""
        # 创建PerformanceAnalysis实例
        from Performance_Analysis import PerformanceAnalysis
        performance_analyzer = PerformanceAnalysis(self.account)

        # 使用PerformanceAnalysis计算各项指标
        total_return = performance_analyzer.get_total_return()
        annualized_return = performance_analyzer.get_annualized_return()
        sharpe_ratio = performance_analyzer.get_sharpe_ratio()
        trade_count = performance_analyzer.get_trade_count()
        buy_count, sell_count = performance_analyzer.get_buy_sell_count()

        print(f"\n绩效指标:")
        print(f"总收益率: {total_return:.2f}%")
        print(f"年化收益率: {annualized_return:.2f}%")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"交易次数: {trade_count}")

        # 打印交易统计
        print(f"\n交易统计:")
        print(f"买入次数: {buy_count}")
        print(f"卖出次数: {sell_count}")

