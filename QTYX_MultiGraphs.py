#! /usr/bin/env python 
#-*- encoding: utf-8 -*- 
#author 元宵大师 本例程仅用于教学目的，严禁转发和用于盈利目的，违者必究

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import mpl_finance as mpf #替换 import matplotlib.finance as mpf
#import mpl_finance as mpf   #comment by weiyf @20210615 due to using new lib named "mplfinance
import mplfinance as mpf  #add by weiyf @20210615
import matplotlib.gridspec as gridspec#分割子图
import sys

class DefTypesPool():

    def __init__(self):
        self.routes = {}

    def route_types(self, types_str):
        def decorator(f):
            self.routes[types_str] = f
            return f
        return decorator

    def route_output(self, path):
        #print(u"output [%s] function:" % path)
        function_val = self.routes.get(path)
        if function_val:
            return function_val
        else:
            raise ValueError('Route "{}"" has not been registered'.format(path))

class MplTypesDraw():

    mpl = DefTypesPool()

    @mpl.route_types(u"line")
    def line_plot(df_index, df_dat, graph):
        # 绘制line图
        for key, val in df_dat.items():
            graph.plot(np.arange(0, len(val)), val, label=key, lw=1.0)

    @mpl.route_types(u"ochl")
    def ochl_plot(df_index, df_dat, graph):
        # 绘制ochl图——Kline
        #原mpf_finance方法
        """
        # 方案一
        mpf.candlestick2_ochl(graph, df_dat['Open'], df_dat['Close'], df_dat['High'], df_dat['Low'], width=0.5,
                              colorup='r', colordown='g') # 绘制K线走势
        # 方案二
        ohlc = list(zip(np.arange(0,len(df_index)),df_dat['Open'], df_dat['Close'], df_dat['High'], df_dat['Low'])) # 使用zip方法生成数据列表
        mpf.candlestick_ochl(graph, ohlc, width=0.2, colorup='r', colordown='g', alpha=1.0) # 绘制K线走势
        """

        #below is the mpf方法
        """SubModule-1 start:  added by weiyf@06-21."""
        """
        make_marketcolors()   setting K colors
        :up
        :down
        : edge
        : wick
        : volumn
        : inherit
        """
        def_color = mpf.make_marketcolors(up='red', down='green', edge='black', wick='black')
        """
        ake_marketcolors()   setting K colors
                :up
                :down
                : edge
                : wick
                : volumn
                : inherit
         """
        def_style = mpf.make_mpf_style(marketcolors = def_color)
        mpf.plot(pd.DataFrame(df_dat), type='candle', style=def_style, ax=graph)
        """SubModule-1 end:  added by weiyf@06-21."""

    @mpl.route_types(u"bar")
    def bar_plot(df_index, df_dat, graph):
        # 绘制bar图——Volume
        #graph.bar(np.arange(0, len(df_index)), df_dat['Volume'], \
        #         color=['g' if df_dat['Open'][x] > df_dat['Close'][x] else 'r' for x in range(0,len(df_index))])

        graph.bar(np.arange(0, len(df_index)), df_dat['bar_red'], facecolor='red')
        graph.bar(np.arange(0, len(df_index)), df_dat['bar_green'], facecolor='green')

    @mpl.route_types(u"hline")
    def hline_plot(df_index, df_dat, graph):
        # 绘制hline图
        for key, val in df_dat.items():
            graph.axhline(val['pos'], c=val['c'], label=key)

    @mpl.route_types(u"annotate")
    def annotate_plot(df_index, df_dat, graph):
        # 绘制annotate图
        for key, val in df_dat.items():
            for kl_index, today in val['andata'].iterrows():
                x_posit = df_index.get_loc(kl_index)
                graph.annotate(u"{}\n{}".format(key, today.name.strftime("%m.%d")),
                                   xy=(x_posit, today[val['xy_y']]),
                                   xycoords='data',
                                   xytext=(val['xytext'][0], val['xytext'][1]),
                                   va=val['va'],  # 点在标注下方
                                   textcoords='offset points',
                                   fontsize=val['fontsize'],
                                   arrowprops=val['arrow'])

    @mpl.route_types(u"filltrade")
    def filltrade_plot(df_index, df_dat, graph):
        # 绘制filltrade图
        signal_shift = df_dat['signal'].shift(1)
        signal_shift.fillna(value=-1, inplace=True)  # 序列最前面的NaN值用-1填充
        list_signal = np.sign(df_dat['signal'] - signal_shift)
        bs_singal = list_signal[list_signal != 0]

        skip_days = False
        for kl_index, value in bs_singal.iteritems(): # iteritems以迭代器形式返回
            if (value == 1) and (skip_days == False) :
                start = df_index.get_loc(kl_index)
                skip_days = True
            elif (value == -1) and (skip_days == True) :
                end = df_index.get_loc(kl_index) + 1  # 加1用于匹配[start:end]选取到end值
                skip_days = False

                if df_dat['jdval'][end-1] < df_dat['jdval'][start]: # 赔钱显示绿色
                    graph.fill_between(np.arange(start, end), 0, df_dat['jdval'][start:end], color='green', alpha=0.38)
                    is_win = False
                else:  # 赚钱显示红色
                    graph.fill_between(np.arange(start, end), 0, df_dat['jdval'][start:end], color='red', alpha=0.38)
                    is_win = True
                graph.annotate('获利\n' if is_win else '亏损\n',
                             xy=(end, df_dat['jdval'].asof(kl_index)),
                             xytext=(df_dat['xytext'][0], df_dat['xytext'][1]),
                             xycoords='data',
                             va=df_dat['va'], # 点在标注下方
                             textcoords='offset points',
                             fontsize=df_dat['fontsize'],
                             arrowprops=df_dat['arrow'])
        # 整个时间序列填充为底色blue 透明度alpha小于后标注区间颜色
        graph.fill_between(np.arange(0, len(df_index)), 0, df_dat['jdval'], color='blue', alpha=.08)


class MultiGraphIf(MplTypesDraw):

    app = DefTypesPool()
    ##########################行情分析界面###############################
    @app.route_types(u"ochl")
    def ochl_graph(stock_dat, sub_graph, df_dat=None):  # prepare data
        type_dict = {'Open': stock_dat.Open,
                     'Close': stock_dat.Close,
                     'High': stock_dat.High,
                     'Low': stock_dat.Low
                     }
        view_function = MplTypesDraw.mpl.route_output(u"ochl")
        view_function(stock_dat.index, type_dict, sub_graph)

    @app.route_types(u"sma")
    def sma_graph(stock_dat, sub_graph, periods):  # prepare data
        for val in periods:
            type_dict = {'SMA'+str(val): stock_dat.Close.rolling(window=val).mean()}
            view_function = MplTypesDraw.mpl.route_output(u"line")
            view_function(stock_dat.index, type_dict, sub_graph)

    @app.route_types(u"vol")
    def vol_graph(stock_dat, sub_graph, df_dat=None):  # prepare data

        type_dict = {'bar_red': np.where(stock_dat.Open < stock_dat.Close, stock_dat.Volume, 0), # 绘制BAR>0 柱状图
                     'bar_green': np.where(stock_dat.Open > stock_dat.Close, stock_dat.Volume, 0) # 绘制BAR<0 柱状图
                     }
        view_function = MplTypesDraw.mpl.route_output(u"bar")
        view_function(stock_dat.index, type_dict, sub_graph)


    @app.route_types(u"macd")
    def macd_graph(stock_dat, sub_graph, df_dat=None):  # prepare data

        macd_dif = stock_dat['Close'].ewm(span=12, adjust=False).mean() - stock_dat['Close'].ewm(span=26,                                                                                         adjust=False).mean()
        macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
        macd_bar = 2 * (macd_dif - macd_dea)

        type_dict = {'bar_red': np.where(macd_bar > 0, macd_bar, 0),  # 绘制BAR>0 柱状图
                     'bar_green': np.where(macd_bar < 0, macd_bar, 0)  # 绘制BAR<0 柱状图
                     }
        view_function = MplTypesDraw.mpl.route_output(u"bar")
        view_function(stock_dat.index, type_dict, sub_graph)

        type_dict = {'macd dif': macd_dif,
                     'macd dea': macd_dea
                     }
        view_function = MplTypesDraw.mpl.route_output(u"line")
        view_function(stock_dat.index, type_dict, sub_graph)

    @app.route_types(u"kdj")
    def kdj_graph(stock_dat, sub_graph, df_dat=None):  # prepare data

        low_list = stock_dat['Low'].rolling(9, min_periods=1).min()
        high_list = stock_dat['High'].rolling(9, min_periods=1).max()
        rsv = (stock_dat['Close'] - low_list) / (high_list - low_list) * 100
        stock_dat['K'] = rsv.ewm(com=2, adjust=False).mean()
        stock_dat['D'] = stock_dat['K'].ewm(com=2, adjust=False).mean()
        stock_dat['J'] = 3 * stock_dat['K'] - 2 * stock_dat['D']

        type_dict = {'K': stock_dat.K,
                     'D': stock_dat.D,
                     'J': stock_dat.J
                    }
        view_function = MplTypesDraw.mpl.route_output(u"line")
        view_function(stock_dat.index, type_dict, sub_graph)


    ##########################回测分析界面###############################
    @app.route_types(u"cash_profit")# cash profit and retracement
    def cash_profit_graph(stock_dat, sub_graph, para_dat = None):

        # 回测参数 由GUI配置
        cash_hold = float(para_dat["cash_hold"]) # cash_hold = 100000
        slippage = float(para_dat["slippage"]) # slippage = 0.01
        c_rate = float(para_dat["c_rate"]) # c_rate = 5.0 / 10000
        t_rate = float(para_dat["t_rate"]) # t_rate = 1.0 / 1000
        stake_size = para_dat["stake_size"] # 买入规模

        posit_num = 0  # 持股数目
        skip_days = False  # 持股/持币状态

        # 最大风险回撤——资金最大回撤
        # 绝对收益—资金的度量
        for kl_index, today in stock_dat.iterrows():
            # 买入/卖出执行代码
            if today.Signal == 1 and skip_days == False:  # 买入
                if stake_size == 'all': # 全仓买入
                    posit_num = int(cash_hold / (today.Close + slippage))  # 资金转化为股票
                    posit_num = int(posit_num / 100) * 100  # 买入股票最少100股，对posit_num向下取整百
                else: # 定额买入
                    posit_num = int(stake_size)
                buy_cash = posit_num * (today.Close + slippage)  # 计算买入股票所需现金
                # 计算手续费，不足5元按5元收，并保留2位小数
                commission = round(max(buy_cash * c_rate, 5), 2)

                if cash_hold > buy_cash - commission :
                    cash_hold = cash_hold - buy_cash - commission
                    skip_days = True # 买入成功
                    print("{0} 成功买入{1}股".format(kl_index.strftime("%y-%m-%d"), posit_num))
                else:
                    print("{0} 您当前的资金不足以买入{1}股".format(kl_index.strftime("%y-%m-%d"), posit_num))

            elif today.Signal == -1 and skip_days == True:  # 卖出 避免未买先卖
                skip_days = False
                sell_cash = posit_num * (today.Close - slippage) # 计算卖出股票得到的现金 卖出股票可以不是整百。
                # 计算手续费，不足5元按5元收，并保留2位小数
                commission = round(max(sell_cash * c_rate, 5), 2)
                # 计算印花税，保留2位小数
                tax = round(sell_cash * t_rate, 2)
                cash_hold = cash_hold + sell_cash - commission - tax # 剩余现金
                print("{0} 成功卖出{1}股".format(kl_index.strftime("%y-%m-%d"), posit_num))

            if skip_days == True:  # 持股
                stock_dat.loc[kl_index, 'total'] = posit_num * today.Close + cash_hold
            else:  # 空仓
                stock_dat.loc[kl_index, 'total'] = cash_hold

            # expanding()计算资金曲线当前的滚动最高值
            stock_dat['max_total'] = stock_dat['total'].expanding().max()
            # 计算资金曲线在滚动最高值之后所回撤的百分比
            stock_dat['per_total'] = stock_dat['total'] / stock_dat['max_total']

        min_point_df = stock_dat.sort_values(by=['per_total'])[0:1]
        min_point_total = min_point_df.per_total
        max_point_df = stock_dat[stock_dat.index <= min_point_total.index[0]].sort_values(by=['total'],
                                                                                          ascending=False)[0:1]
        max_point_total = max_point_df.total
        # min_point_total = stock_dat.sort_values(by=['per_total']).iloc[[0], stock_dat.columns.get_loc('per_total')]
        # max_point_total = stock_dat[stock_dat.index <= min_point_total.index[0]].sort_values \
        #    (by=['total'], ascending=False).iloc[[0], stock_dat.columns.get_loc('total')]

        print("资金最大回撤%5.2f%% 从%s开始至%s结束" % ((1 - min_point_total.values) * 100, \
                                            max_point_total.index[0], min_point_total.index[0]))
        # 最大资金回撤 7.53%从2018-07-13 00:00:00开始至2018-09-12 00:00:00结束

        line_total = "资金总体收益%d；上涨幅度 %.2f%%" % (stock_dat['total'][-1], (stock_dat['total'][-1] - 100000) / 100000 * 100)
        print(line_total)
        max_total = "资金滚动最高值"

        type_dict = {line_total: stock_dat.total,
                     max_total: stock_dat.max_total,
                     }
        view_function = MplTypesDraw.mpl.route_output(u"line")
        view_function(stock_dat.index, type_dict, sub_graph)

        type_dict = {u"资金最大回撤\n{}".format(1 - min_point_total.values):
                         {'andata': min_point_df,
                          'va': 'top',
                          'xy_y': 'total',
                          'xytext': (0,stock_dat['High'].mean()),
                          'fontsize': 8,
                          'arrow': dict(facecolor='green', shrink=0.1)
                          },
                      }
        view_function = MplTypesDraw.mpl.route_output(u"annotate")
        view_function(stock_dat.index, type_dict, sub_graph)

    @app.route_types(u"cmp_profit")  # relative_profit
    def cmp_profit_graph(stock_dat, sub_graph, para_dat):

        # 相对收益—策略VS基准
        stock_dat['benchmark_profit_log'] = np.log(stock_dat.Close / stock_dat.Close.shift(1))
        stock_dat.loc[stock_dat.Signal == -1, 'Signal'] = 0
        stock_dat['trend_profit_log'] = stock_dat['Signal'] * stock_dat.benchmark_profit_log
        line_trend_key = "策略收益%.2f" % stock_dat['trend_profit_log'].cumsum()[-1]
        line_bench_key = "基准收益%.2f" % stock_dat['benchmark_profit_log'].cumsum()[-1]
        print("资金相对收益：%s VS %s" % (line_trend_key, line_bench_key))

        type_dict = {line_bench_key: stock_dat['benchmark_profit_log'].cumsum(),
                     line_trend_key: stock_dat['trend_profit_log'].cumsum()
                     }
        view_function = MplTypesDraw.mpl.route_output(u"line")
        view_function(stock_dat.index, type_dict, sub_graph)

    @app.route_types(u"close_retrace")  # relative_profit
    def close_retrace_graph(stock_dat, sub_graph, para_dat):
        # 度量策略最大风险回撤——收盘价最大回撤

        # 计算收盘价曲线当前的滚动最高值
        stock_dat['max_close'] = stock_dat['Close'].expanding().max()
        # 计算收盘价曲线在滚动最高值之后所回撤的百分比
        stock_dat['per_close'] = stock_dat['Close'] / stock_dat['max_close']

        # 计算并打印收盘价的最大回撤率
        min_point_df = stock_dat.sort_values(by=['per_close'])[0:1]
        min_point_close = min_point_df.per_close
        # min_point_close = stock_dat.sort_values(by=['per_close']).iloc[[0], stock_dat.columns.get_loc('per_close')]

        # 寻找出最大回撤率所对应的最高值交易日和最大回撤交易日，并打印显示
        max_point_df = stock_dat[stock_dat.index <= min_point_close.index[0]].sort_values(by=['Close'],
                                                                                          ascending=False)[0:1]
        max_point_close = max_point_df.Close
        # max_point_close = stock_dat[stock_dat.index <= min_point_close.index[0]].sort_values \
        #    (by=['Close'], ascending=False).iloc[[0], stock_dat.columns.get_loc('Close')]

        print("股价最大回撤%5.2f%% 从%s开始至%s结束" % ((1 - min_point_close.values) * 100, \
                                            max_point_close.index[0], min_point_close.index[0]))
        ##最大股价回撤 29.21% 从2018-06-12 00:00:00开始至2018-12-27 00:00:00结束

        type_dict = {'最大收盘价': stock_dat.max_close,
                     '收盘价': stock_dat.Close
                     }
        view_function = MplTypesDraw.mpl.route_output(u"line")
        view_function(stock_dat.index, type_dict, sub_graph)

        type_dict = {u"股价最大回撤\n{}".format(1 - min_point_close.values):
                               {'andata': min_point_df,
                                'va': 'top',
                                'xy_y': 'Close',
                                'xytext': (0,stock_dat['High'].mean()),
                                'fontsize': 8,
                                'arrow': dict(facecolor='green', shrink=0.1)
                                },
                    }
        view_function = MplTypesDraw.mpl.route_output(u"annotate")
        view_function(stock_dat.index, type_dict, sub_graph)

    @app.route_types(u"trade")
    def trade_graph(stock_dat, sub_graph, para_dat):
        # 交易获利/亏损区间可视化

        type_dict = {'signal': stock_dat.Signal,
                     'jdval': stock_dat.Close,
                     'va': 'top',
                     'xy_y': 'Close',
                     'xytext': (0,stock_dat['High'].mean()),
                     'fontsize': 8,
                     'arrow': dict(facecolor='yellow', shrink=0.1)
                     }
        view_function = MplTypesDraw.mpl.route_output(u"filltrade")
        view_function(stock_dat.index, type_dict, sub_graph)

    def __init__(self, **kwargs):
        MplTypesDraw.__init__(self)
        self.fig = plt.figure(figsize=kwargs['figsize'], dpi=100, facecolor="white")#创建fig对象
        self.graph_dict = {}
        self.graph_curr = []

        try:
            gs = gridspec.GridSpec(kwargs['nrows'], kwargs['ncols'],
                                   left = kwargs['left'], bottom = kwargs['bottom'], right = kwargs['right'], top = kwargs['top'],
                                   wspace = kwargs['wspace'], hspace = kwargs['hspace'], height_ratios = kwargs['height_ratios'])
        except:
            raise Exception("para error")
        else:
            for i in range(0, kwargs['nrows'], 1):
                self.graph_dict[kwargs['subplots'][i]] = self.fig.add_subplot(gs[i, :])

    def log_trade_info(self, stock_dat):

        signal_shift = stock_dat.Signal.shift(1)
        signal_shift.fillna(value=-1, inplace=True)  # 序列最前面的NaN值用-1填充
        list_signal = np.sign(stock_dat.Signal - signal_shift)

        buy_singal = stock_dat[list_signal.isin([1])]
        sell_singal = stock_dat[list_signal.isin([-1])]
        """
        trade_info = pd.DataFrame({'BuyTime': buy_singal.index.strftime("%y.%m.%d"),
                                   'SellTime': sell_singal.index.strftime("%y.%m.%d"),
                                   'BuyPrice': buy_singal.Close.values,
                                   'SellPrice': sell_singal.Close.values})
        """
        # 解决 ValueError: Length of values does not match length of index
        trade_info = pd.concat([pd.DataFrame({'BuyTime': buy_singal.index.strftime("%y.%m.%d")}),
                                pd.DataFrame({'SellTime': sell_singal.index.strftime("%y.%m.%d")}),
                                pd.DataFrame({'BuyPrice': buy_singal.Close.values}),
                                pd.DataFrame({'SellPrice': sell_singal.Close.values})], axis=1)

        trade_info['DiffPrice'] = trade_info.SellPrice - trade_info.BuyPrice
        trade_info['PctProfit'] = np.round(trade_info.DiffPrice / trade_info.BuyPrice*100, 2)

        win_count = (trade_info.DiffPrice >= 0).sum()
        loss_count = (trade_info.DiffPrice < 0).sum()
        win_profit = trade_info[trade_info.PctProfit >= 0].PctProfit.sum()
        loss_profit = trade_info[trade_info.PctProfit < 0].PctProfit.sum()

        for kl_index, info in trade_info.iterrows():
            print(f'策略信号-{kl_index}')
            print(f'买入时间：{info.BuyTime}, 买入价格：{info.BuyPrice}')
            print(f'卖出时间:{info.SellTime}，卖出价格：{info.SellPrice}')
            print(f'涨幅%：{info.PctProfit}')

        print(f'亏损次数:{loss_count}, 盈利次数:{win_count}, 胜率:{round(win_count / (win_count + loss_count)*100, 2)}%')
        print(f'平均亏损:{round((loss_profit / loss_count), 2)}% 平均盈利:{round((win_profit / win_count), 2)}%')

    def graph_attr(self, **kwargs):

        if 'title' in kwargs.keys():
            self.graph_curr.set_title(kwargs['title'])

        if 'legend' in kwargs.keys():
            self.graph_curr.legend(loc=kwargs['legend'], shadow=True)

        if 'xlabel' in kwargs.keys():
            self.graph_curr.set_xlabel(kwargs['xlabel'])

        self.graph_curr.set_ylabel(kwargs['ylabel'])
        self.graph_curr.set_xlim(0, len(self.df_ohlc.index))  # 设置一下x轴的范围
        self.graph_curr.set_xticks(range(0, len(self.df_ohlc.index), kwargs['xticks']))  # X轴刻度设定 每15天标一个日期

        if 'xticklabels' in kwargs.keys():
            self.graph_curr.set_xticklabels(
                [self.df_ohlc.index.strftime(kwargs['xticklabels'])[index] for index in
                 self.graph_curr.get_xticks()])  # 标签设置为日期

            # X-轴每个ticker标签都向右倾斜45度
            for label in self.graph_curr.xaxis.get_ticklabels():
                label.set_rotation(45)
                label.set_fontsize(10)  # 设置标签字体
        else:
            for label in self.graph_curr.xaxis.get_ticklabels():
                label.set_visible(False)

class Sys_MultiGraph(MultiGraphIf):

    # 参考第9章
    def back_graph_run(self, stock_data, **kwargs):
        # 绘制子图
        self.df_ohlc = stock_data

        # 临时把标准输出重定向到一个文件，然后再恢复正常
        with open('logtrade.txt', 'w') as f:
            oldstdout = sys.stdout
            sys.stdout = f
            try:
                self.log_trade_info(self.df_ohlc)
                for key in kwargs:
                    self.graph_curr = self.graph_dict[kwargs[key]['graph_name']]
                    for path, val in kwargs[key]['graph_type'].items():
                        view_function = MultiGraphIf.app.route_output(path)
                        view_function(self.df_ohlc, self.graph_curr, val)
                    self.graph_attr(**kwargs[key])
                #plt.show()
            finally:
                sys.stdout = oldstdout

        """
        print("kwargs %s-->%s" % (key, kwargs[key]))
        #globals().get('self.%s' % key)(**kwargs[key])
        eval('self.%s' % key)()
        #self.kline_draw(**kwargs[key])
        """

    # 参考第8章
    def firm_graph_run(self, stock_data, **kwargs):
        # 绘制子图
        self.df_ohlc = stock_data
        for key in kwargs:
            self.graph_curr = self.graph_dict[kwargs[key]['graph_name']]
            for path, val in kwargs[key]['graph_type'].items():
                view_function = MultiGraphIf.app.route_output(path)
                view_function(self.df_ohlc, self.graph_curr, val)
            self.graph_attr(**kwargs[key])
        #plt.show()