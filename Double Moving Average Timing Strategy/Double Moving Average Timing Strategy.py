#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:04:19 2019

@author: x.yi
"""
'''
Double Moving Average Timing Strategy
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#%%
def strategy(data, win_long, win_short, loss_ratio=0.99):
    '''
    parameters
        win_long: rolling windows for long period
        win_short: rolloing windows for short period
        loss_ratio: maximum loss ratio we can afford; if exceeded, close position
    '''
    datas = data.copy()
    # compute long and short moving average 
    datas['longma'] = datas.CLOSE.rolling(win_long,min_periods=0).mean()
    datas['shortma'] = datas.CLOSE.rolling(win_short,min_periods=0).mean()
     
    datas['position'] = 0 # record position
    datas['flag'] = 0     # record transactions
     
    pricein = []
    priceout = []
    price_in = 1
    for i in range(max(1,win_long),datas.shape[0]-1):
        # if position = 0 now, and short moving average shadows long  
        # moving average, then long
        if (datas.shortma[i-1] < datas.longma[i-1]) & (datas.shortma[i] >  \
            datas.longma[i]) & (datas.position[i] == 0):
                 
            datas.loc[i,'flag'] = 1
            datas.loc[i+1,'position'] = 1
            date_in = datas.DateTime[i]
            price_in = datas.CLOSE[i]
            pricein.append([date_in,price_in])
        # if position = 1 now, and drop down exceeds max loss ratio, then stop loss
        elif (datas.position[i] == 1) & (datas.CLOSE[i]/price_in < loss_ratio):
             
            datas.loc[i,'flag'] = -1
            datas.loc[i+1,'position'] = 0
            date_out = datas.DateTime[i]
            price_out = datas.CLOSE[i]
            priceout.append([date_out,price_out])
        # if position = 1 now, and short moving average passes long moving \
        # average from top to bottom, then close the position
        elif (datas.position[i] == 1) & (datas.shortma[i-1] > datas.longma[i-1]) & \
             (datas.shortma[i] < datas.longma[i]):
              
            datas.loc[i,'flag'] = -1
            datas.loc[i+1,'position'] = 0
            date_out = datas.DateTime[i]
            price_out = datas.CLOSE[i]
            priceout.append([date_out,price_out])
         # otherwise, keep current position
        else:
            datas.loc[i+1,'position'] = datas.loc[i,'position']
    # integrate buy information and sell information  
    buyinfo = pd.DataFrame(pricein,columns=['buydate','buyprice'])
    sellinfo = pd.DataFrame(priceout,columns=['selldate','sellprice'])
    # integrate transaction information
    transaction = pd.concat([buyinfo,sellinfo], axis=1)
     
    datas = datas.loc[max(0,win_long):,:].reset_index(drop=True)
    datas['returns'] = datas.CLOSE.pct_change(1).fillna(0)
    datas['cumulative_ret'] = (1+datas.returns * datas.position).cumprod()
    datas['benchmark'] = datas.CLOSE/datas.CLOSE[0]
     
    return datas, transaction

#%%

def performance(transaction, strategy):
    '''
    parameters
        transaction: transaction information we got from 'strategy' function
        strategy: strategy information we got from 'strategy' function
    '''
    N = 252 # 252 trading days in one year
    # compute annualized return
    annual_return = strategy.cumulative_ret[strategy.shape[0]-1] ** (N/strategy.shape[0]) - 1
    # compute Sharpe Ratio
    sharpe = (strategy.returns * strategy.position).mean() /  \
             (strategy.returns * strategy.position).std() * np.sqrt(N)
    # compute victory ratios
    victory_ratio = (transaction.sellprice > transaction.buyprice).mean()
    # compute maximun drop down
    maxdropdown = (1-strategy.cumulative_ret / strategy.cumulative_ret.cummax()).max()
    # compute max loss for every trading
    maxloss = min(transaction.sellprice / transaction.buyprice-1) 
    
    # compute annulized performance of strategy
    strategy['year'] = strategy.DateTime.apply(lambda x:x[0:4])
    # compute cumulative return per year
    cumulative_ret_peryear = strategy.cumulative_ret.groupby(strategy.year).last() \
            / strategy.cumulative_ret.groupby(strategy.year).first() -1
    # compute benchmark return per year
    benchmark_peryear = strategy.benchmark.groupby(strategy.year).last() / \
                strategy.benchmark.groupby(strategy.year).first() - 1
    # compute excessive return per year
    excess_peryear = cumulative_ret_peryear - benchmark_peryear
    # concat the three returns
    result_peryear = pd.concat([cumulative_ret_peryear,benchmark_peryear, \
                                excess_peryear], axis = 1)
    result_peryear.columns = ['strategy_return','benchmark_return','excessive_return']  
    
    
    strategy.set_index('DateTime',inplace=True)
    strategy.index = pd.to_datetime(strategy.index)
    # plot the result
    plt.rcParams['figure.figsize'] = (14,12)
    plt.subplot(211)
    plt.plot(strategy.cumulative_ret,label='strategy')
    plt.plot(strategy.benchmark,label='benchmark')
    plt.legend(loc='upper left',fontsize=15)
    plt.title('Cumulative returns',fontsize=20)
    plt.subplot(212)
    plt.plot(result_peryear.strategy_return,label='strategy')
    plt.plot(result_peryear.benchmark_return,label='benchmark')
    plt.plot(result_peryear.excessive_return,label='excessive')
    plt.legend(loc='upper left',fontsize=15)
    plt.title('Yearly cumulative return',fontsize=20)
    
    
    
    print('-------------------------------------------')
    print('Sharpe Ratio:',round(sharpe,2))
    print('Annualized return:{}%'.format(round(annual_return*100,2)))
    print('Victory Ratio:{}%'.format(round(victory_ratio*100,2)))
    print('Maximum drop down:{}%'.format(round(maxdropdown,2)))
    print('Maximun loss for one single trading:{}%'.format(round(-maxloss*100,2)))
    print('Monthly average trading times:{}'.format(round( \
          strategy.flag.abs().sum()/strategy.shape[0]*30,2)))
    
    result = {'Sharpe_ratio':sharpe,
              'Annualized_return':annual_return,
              'Victory_ratio':victory_ratio,
              'Max_drop_down':maxdropdown,
              'MaxlossOnce':-1*maxloss,
              'Average_trade_num':round(strategy.flag.abs().sum()/strategy.shape[0],1)
              }
    result = pd.DataFrame.from_dict(result,orient='index').T
    
    return result, result_peryear

#%%
# applied to CSI 300 index from 2010-1-4 to 2019-3-15
data = pd.read_csv('沪深300.csv')
win_long = 26
win_short = 12
strategyres, transaction = strategy(data,win_long,win_short)
finalresult, yearlyresult = performance(transaction,strategyres)

#%%















        