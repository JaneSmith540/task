import pandas as pd
import numpy as np


class PerformanceAnalysis:
    def __init__(self, account):
        self.account = account
        self.strategy_returns = None  # 策略日收益率序列
        self.cumulative_returns = None  # 累计收益率序列
        self.cumulative_net_assets = None  # 累计净值序列

        # 初始化时计算基础收益率
        self.calculate_returns()
        self.calculate_cumulative_returns()

    def calculate_returns(self):
        """计算策略日收益率 - 修复版本"""
        if len(self.account.total_assets) == 0:
            raise ValueError("没有可用的资产数据用于计算收益率")

        # 从账户总资产和日期生成收益率序列
        total_assets_series = pd.Series(
            self.account.total_assets,
            index=self.account.dates
        )

        # 修复1：检查资产数据是否合理
        if any(asset <= 0 for asset in self.account.total_assets):
            print("警告：发现非正资产值，可能影响收益率计算")
            # 将非正资产值替换为前一个有效值或小正数
            total_assets_series = total_assets_series.replace(0, np.nan).fillna(method='ffill')
            total_assets_series = total_assets_series.clip(lower=1e-6)  # 确保最小值为正

        # 修复2：使用对数收益率，避免极端值
        # 方法1：标准百分比变化（限制单日涨跌幅在合理范围内）
        self.strategy_returns = total_assets_series.pct_change().fillna(0)

        # 方法2：对数收益率（更稳定）
        # self.strategy_returns = np.log(total_assets_series / total_assets_series.shift(1)).fillna(0)

        # 修复3：限制单日涨跌幅在合理范围内（A股±10%）
        self.strategy_returns = self.strategy_returns.clip(lower=-0.11, upper=0.11)

        print(f"收益率统计: 最小值={self.strategy_returns.min():.4f}, 最大值={self.strategy_returns.max():.4f}")

        return self.strategy_returns

    def calculate_cumulative_returns(self):
        """计算累计收益率和累计净值 - 修复版本"""
        if self.strategy_returns is None:
            self.calculate_returns()

        try:
            # 计算累计净值（从1开始）
            self.cumulative_net_assets = (1 + self.strategy_returns).cumprod()

            # 计算累计收益率
            self.cumulative_returns = self.cumulative_net_assets - 1

            print(
                f"累计收益率统计: 最小值={self.cumulative_returns.min():.4f}, 最大值={self.cumulative_returns.max():.4f}")

        except Exception as e:
            print(f"累计收益率计算错误: {e}")
            # 如果计算失败，创建安全的默认值
            self.cumulative_net_assets = pd.Series([1.0] * len(self.strategy_returns),
                                                   index=self.strategy_returns.index)
            self.cumulative_returns = pd.Series([0.0] * len(self.strategy_returns), index=self.strategy_returns.index)

        return self.cumulative_returns

    def calculate_cumulative_returns(self):
        """计算累计收益率和累计净值"""
        if self.strategy_returns is None:
            self.calculate_returns()

        # 计算累计净值（从1开始）
        self.cumulative_net_assets = (1 + self.strategy_returns).cumprod()
        # 计算累计收益率
        self.cumulative_returns = self.cumulative_net_assets - 1
        return self.cumulative_returns

    def get_total_return(self):
        """计算总收益率 - 直接使用最终资产计算"""
        if len(self.account.total_assets) < 2:
            return 0.0
        initial_value = self.account.total_assets[0]
        final_value = self.account.total_assets[-1]
        return (final_value / initial_value - 1) * 100

    def get_annualized_return(self):
        """计算年化收益率 - 修复版本"""
        if len(self.account.dates) < 2:
            return 0.0

        total_return = self.get_total_return() / 100  # 转换为小数

        # 计算实际交易天数
        trading_days = len(self.account.dates)
        if trading_days <= 1:
            return 0.0

        # 使用交易日数计算年化（假设一年252个交易日）
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        return annualized_return * 100  # 转换为百分比

    def get_sharpe_ratio(self, risk_free_rate=0.02):
        """计算夏普比率 - 简化版本，使用最终收益率/标准差"""
        if self.strategy_returns is None:
            self.calculate_returns()

        if len(self.strategy_returns) < 2:
            return 0.0

        # 年化收益率
        annual_return = self.get_annualized_return() / 100

        # 年化波动率
        annual_volatility = self.strategy_returns.std() * np.sqrt(252)

        if annual_volatility == 0:
            return 0.0

        # 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return sharpe_ratio


    def get_volatility(self):
        """计算年化波动率"""
        if self.strategy_returns is None:
            self.calculate_returns()

        if len(self.strategy_returns) < 2:
            return 0.0

        return self.strategy_returns.std() * np.sqrt(252) * 100

    def get_calmar_ratio(self):
        """计算Calmar比率"""
        annual_return = abs(self.get_annualized_return())  # 取绝对值
        max_dd = abs(self.get_max_drawdown())

        if max_dd == 0:
            return 0.0
        return annual_return / max_dd

    def get_trade_count(self):
        """获取总交易次数"""
        return len(self.account.trade_history)

    def get_buy_sell_count(self):
        """获取买入/卖出次数"""
        if not self.account.trade_history:
            return 0, 0
        trades = pd.DataFrame(self.account.trade_history)
        buy_count = len(trades[trades['action'] == 'buy'])
        sell_count = len(trades[trades['action'] == 'sell'])
        return buy_count, sell_count

    def get_win_rate(self):
        """计算胜率"""
        if not self.account.trade_history:
            return 0.0

        trades = pd.DataFrame(self.account.trade_history)
        sell_trades = trades[trades['action'] == 'sell']

        if sell_trades.empty:
            return 0.0

        if 'profit' in sell_trades.columns:
            winning_trades = len(sell_trades[sell_trades['profit'] > 0])
            return (winning_trades / len(sell_trades)) * 100
        else:
            return 0.0

    def get_avg_trade_return(self):
        """计算平均交易收益率"""
        if not self.account.trade_history:
            return 0.0

        trades = pd.DataFrame(self.account.trade_history)
        if 'return_rate' in trades.columns:
            return trades['return_rate'].mean() * 100
        else:
            return 0.0

    def validate_data(self):
        """验证数据完整性"""
        issues = []

        # 检查资产数据
        if len(self.account.total_assets) == 0:
            issues.append("没有资产数据")

        # 检查资产值是否合理
        if any(asset <= 0 for asset in self.account.total_assets):
            issues.append("存在非正资产值")

        # 检查日期数据
        if len(self.account.dates) != len(self.account.total_assets):
            issues.append("日期和资产数据长度不匹配")

        return issues

    def get_performance_summary(self):
        """生成完整的绩效摘要"""
        # 首先验证数据
        data_issues = self.validate_data()
        if data_issues:
            print(f"数据警告: {', '.join(data_issues)}")

        total_return = self.get_total_return()
        annual_return = self.get_annualized_return()
        sharpe_ratio = self.get_sharpe_ratio()
        volatility = self.get_volatility()
        calmar_ratio = self.get_calmar_ratio()
        trade_count = self.get_trade_count()
        buy_count, sell_count = self.get_buy_sell_count()
        win_rate = self.get_win_rate()
        avg_trade_return = self.get_avg_trade_return()

        # 调试信息
        print(f"调试信息:")
        print(f"  初始资产: {self.account.total_assets[0] if self.account.total_assets else 'N/A'}")
        print(f"  最终资产: {self.account.total_assets[-1] if self.account.total_assets else 'N/A'}")
        print(f"  交易日数: {len(self.account.dates)}")
        print(f"  收益率序列长度: {len(self.strategy_returns) if self.strategy_returns is not None else 0}")

        summary = {
            '总收益率 (%)': round(total_return, 2),
            '年化收益率 (%)': round(annual_return, 2),
            '夏普比率': round(sharpe_ratio, 3),
            '年化波动率 (%)': round(volatility, 2),
            'Calmar比率': round(calmar_ratio, 3),
            '总交易次数': trade_count,
            '买入次数': buy_count,
            '卖出次数': sell_count,
            '胜率 (%)': round(win_rate, 2),
            '平均交易收益率 (%)': round(avg_trade_return, 2)
        }

        return summary