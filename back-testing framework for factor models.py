#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:59:08 2019

@author: yi
"""
'''
This is a back-testing framework used for factor models
'''
#import neccessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#%%
#def some functions that will be used when show the cumulative returns
'''Assume we have data of factors for each stock everyday whose format is MultiIndex 
   and the data of Prices, Trade Status, Max up or dowm'''
def compute_forwardreturns(prices,period):
    '''compute n days forward returns given every day prices and time period'''
    forward_returns = pd.DataFrame(index=pd.MultiIndex. \
                      from_product([prices.index,prices.columns], \
                      names=['date','asset']))
    delta = prices.pct_change(period).shift(-period)
    forward_returns[period] = delta.stack()
    forward_returns.index = forward_returns.index.rename(['date', 'asset'])
    return forward_returns

def quantize_factor(factor_data,quantiles):
    '''Score the factors by quantizing them given quantiles group by date'''
    def quantile_calc(x, _quantiles):
        return pd.qcut(x, _quantiles, labels=False) + 1
    grouper = [factor_data.index.get_level_values('date')]
    factor_quantile = factor_data.groupby(grouper)['factor'] \
        .apply(quantile_calc, quantiles)   
    return factor_quantile.dropna()
    
def get_factorscore_forwardreturns_weights(factor,prices,quantiles,period, \
                                           is_long_short=True):
    '''weight the stocks by their factor values given whether we want a 
    long-short-position or long-position only'''
    if factor.index.levels[0].tz!=prices.index.tz:
        raise Exception('timezone is not the same')
    merge_data = compute_forwardreturns(prices,period)
    factor = factor.copy()
    factor.index = factor.index.rename(['date','asset'])
    merge_data['factor'] = factor
    merge_data = merge_data.dropna(axis=0,how='any')
    merge_data['factor_quantile'] = quantize_factor(merge_data,quantiles)
      
    def to_weights(group, is_long_short):
        if is_long_short:
            demeaned_vals = group - group.mean()
            return demeaned_vals / demeaned_vals.abs().sum()
        else:
            return group / group.abs().sum()
    grouper = [merge_data.index.get_level_values('date')]
    merge_data['weights'] = merge_data.groupby(grouper)['factor'] \
        .apply(to_weights, is_long_short)
    merge_data = merge_data.dropna(axis=0,how='any')
    
    return merge_data

def factor_returns(factor_datas,period):
    '''compute returns based on the weight we just calculated'''
    weighted_returns = factor_datas[period].multiply(factor_datas['weights'], \
                                  axis=0)
    returns = weighted_returns.groupby(level='date').sum()                                               
    returns = pd.DataFrame(returns)
    return returns

def universe_returns(factor_data, period, demeaned=True):
    '''compute universe returns based on giving each stock equal positon'''
    returns1 = factor_returns(factor_data, period)
    universe_ret = factor_data.groupby(level='date')[period].mean() \
                    .loc[returns1.index]
    universe_ret = pd.DataFrame(universe_ret)
    return universe_ret

def cumulative_returns(returnss,period):
    '''compute cumulative returns given the frequency of position adjustment 
       period we choose'''
    returnss = returnss.fillna(0)
    if period==1:
        return returnss.add(1).cumprod()
    
    def split_portfolio(ret, period): return pd.DataFrame(np.diag(ret))

    sub_portfolios = returnss.groupby(np.arange(len(returnss.index)) // period,
                                     axis=0).apply(split_portfolio, period)
    sub_portfolios.index = returnss.index

    def rate_of_returns(ret, period): 
        return ((np.nansum(ret) + 1)**(1. / period)) - 1

    sub_portfolios = sub_portfolios.rolling(window=period, min_periods=1) \
                                   .apply(rate_of_returns, args=(period,))
    sub_portfolios = sub_portfolios.add(1).cumprod()
    
    return sub_portfolios[0]


def show_uni_vs_fac_cumret(universe_returnss,factor_returnss,period):
    '''compare universe return with factor model return by plotting the 
       cumulative return curve'''
    factor_cumret = cumulative_returns(factor_returnss,period)
    universe_cumret = cumulative_returns(universe_returnss,period)
    plt.plot(universe_cumret,color='blue',label='universe_returns')
    plt.plot(factor_cumret,color='orange',label='factor_returns')
    plt.legend(loc='upper left')
    print('累计收益率：')
    plt.show()
    
'''
def get_top_and_low_npercent(factor_data,largest_q):

    factor_data_onlytoplow = factor_data.copy()
    factor_data_onlytoplow['factor_quantile'] = \
        factor_data_onlytoplow['factor_quantile'] \
        .apply(lambda x: x if x in [1,largest_q] else np.nan)
    factor_data_onlytoplow.loc[factor_data_onlytoplow. \
                               factor_quantile.isnull()==True,'weights']=0  
    
    grouper = [factor_data.index.get_level_values('date')] 
    def adjust_weights(group):
        return group / group.abs().sum()   
    factor_data_onlytoplow['weights'] = \
        factor_data_onlytoplow.groupby(grouper)['weights']. \
        apply(adjust_weights)                  
    return factor_data_onlytoplow
'''

def consider_tradestatus_and_maxupordown(factor_data1,trade_status, \
                                         maxupordown,prices,period):
    '''consider real world situations: trading status and max up or down'''
    def adjust_factor_returns(factor_data1,trade_status,prices,period):
        factor_data_adjust = factor_data1.copy()
        factor_data_adjust_returns = factor_data_adjust[period].unstack()
        factor_data_adjust_returns.index = \
            factor_data_adjust_returns.index.strftime("%Y-%m-%d %H:%M:%S")
        prices1 = prices.copy()
        prices1.index = prices1.index.strftime("%Y-%m-%d %H:%M:%S")
        temp_status = (trade_status.copy()).shift(-period)    
         
        for i in factor_data_adjust_returns.index:
            for j in factor_data_adjust_returns.columns:
                if j in trade_status.columns and \
                    pd.isnull(factor_data_adjust_returns.at[i,j])==False:
                    if trade_status.at[i,j] in ['停牌一天','下午停牌']:
                        factor_data_adjust_returns.at[i,j]=0
                    elif trade_status.at[i,j] not in ['停牌一天','下午停牌'] \
                        and temp_status.at[i,j] in ['停牌一天','下午停牌']:
                        mindate = trade_status[i:] \
                            [period+1:][(trade_status[j]!='停牌一天') \
                              &(trade_status[j]!='下午停牌')].index.tolist()
                        if len(mindate):
                            mindate = mindate[0]
                            factor_data_adjust_returns.at[i,j]= \
                            (prices1.at[mindate,j]-prices1.at[i,j]) \
                                /prices1.at[i,j]
                    else:
                        factor_data_adjust_returns.at[i,j] = \
                            factor_data_adjust_returns.at[i,j]
        factor_data_adjust_returns = factor_data_adjust_returns.stack()
           
        return factor_data_adjust_returns

    factor_data_adjust1 = adjust_factor_returns(factor_data1,trade_status, \
                                                prices,period)
    
    def adjust_factor_returns2(factor_data_adjust1,maxupordown,prices,period):
        factor_data_adjust1 = pd.DataFrame(factor_data_adjust1)
        factor_data_adjust2 = factor_data_adjust1.copy()
        factor_data_adjust2 = factor_data_adjust2[0].unstack()
        prices2 = prices.copy()
        prices2.index = prices2.index.strftime("%Y-%m-%d %H:%M:%S")
        temp_status2 = (maxupordown.copy()).shift(-period)    
         
        for i in factor_data_adjust2.index:
            for j in factor_data_adjust2.columns:
                if j in maxupordown.columns and \
                    pd.isnull(factor_data_adjust2.at[i,j])==False:
                    if maxupordown.at[i,j]==1:
                        factor_data_adjust2.at[i,j]=0
                    elif (maxupordown.at[i,j]==0) and \
                        (temp_status2.at[i,j]==-1):
                        mindate2 = maxupordown[i:] \
                            [period+1:][(maxupordown[j]!=-1)].index.tolist()
                        mindate2 = mindate2[0]
                        factor_data_adjust2.at[i,j]= \
                            (prices2.at[mindate2,j]-prices2.at[i,j]) \
                            /prices2.at[i,j]
                    else:
                        factor_data_adjust2.at[i,j] = \
                            factor_data_adjust2.at[i,j]
        factor_data_adjust2 = factor_data_adjust2.stack()
           
        return factor_data_adjust2
    
    factor_data_adjust2 = adjust_factor_returns2(factor_data_adjust1, 
                                                 maxupordown,prices,period)

    return factor_data_adjust2

def fitin(factor_data1,factor_data_adjust2,period):  
    factor_data1 = pd.DataFrame(factor_data1)
    tempa = factor_data1.copy()
    tempa = tempa[period].unstack()
    tempb = factor_data_adjust2.copy()
    tempb = pd.DataFrame(tempb)
    tempb = tempb[0].unstack()

    for i in range(tempa.shape[0]):
        for j in range(tempa.shape[1]):
            if pd.isnull(tempa.iat[i,j])==True:
                tempb.iat[i,j]=tempa.iat[i,j]
                
    tempb.index = pd.to_datetime(tempb.index)
    tempb = tempb.stack()
    tempb.index = tempb.index.rename(['date','asset'])
    tempa = factor_data1.copy()
    tempa[period] = tempb
    factor_data_new = tempa.copy()
    
    return factor_data_new

def consider_tradestatus_and_maxupordown2(factor_data2,trade_status, \
                                          maxupordown,prices,period):
    def adjust_factor_weights(factor_data2,trade_status,prices,period):
        factor_data_weight = factor_data2.copy()
        factor_data_weight = factor_data_weight['weights'].unstack()
        factor_data_weight.index = factor_data_weight.index \
            .strftime("%Y-%m-%d %H:%M:%S")
        prices3 = prices.copy()
        prices3.index = prices3.index.strftime("%Y-%m-%d %H:%M:%S")
            
        for i in factor_data_weight.index:
            for j in factor_data_weight.columns:
                if j in trade_status.columns and \
                    pd.isnull(factor_data_weight.at[i,j])==False:
                    if trade_status.at[i,j] in ['停牌一天','下午停牌']:
                        factor_data_weight.at[i,j]=0   
                    else:                    
                        factor_data_weight.at[i,j] = factor_data_weight.at[i,j]
                        
        factor_data_weight = factor_data_weight.stack()
           
        return factor_data_weight
    
    factor_data_adjust3 = adjust_factor_weights(factor_data2,trade_status, \
                                                prices,period)

    def adjust_factor_weights2(factor_data_adjust3,maxupordown,prices,period):
        factor_data_adjust3 = pd.DataFrame(factor_data_adjust3)
        factor_data_weight2 = factor_data_adjust3.copy()
        factor_data_weight2 = factor_data_weight2[0].unstack()
        prices4 = prices.copy()
        prices4.index = prices4.index.strftime("%Y-%m-%d %H:%M:%S")
            
        for i in factor_data_weight2.index:
            for j in factor_data_weight2.columns:
                if j in maxupordown.columns and \
                    pd.isnull(factor_data_weight2.at[i,j])==False:
                    if maxupordown.at[i,j]==1 and factor_data_weight2.at[i,j]>0:                        
                        factor_data_weight2.at[i,j]=0
                    elif maxupordown.at[i,j]==-1 and \
                        factor_data_weight2.at[i,j]<0:
                        factor_data_weight2.at[i,j]=0
                    else:
                        factor_data_weight2.at[i,j]=factor_data_weight2.at[i,j]
                        
        factor_data_weight2 = factor_data_weight2.stack()
           
        return factor_data_weight2
    
    factor_data_adjust4 = adjust_factor_weights2(factor_data_adjust3, \
                                                 maxupordown,prices,period)
    
    return factor_data_adjust4

def fitin2(factor_data2,factor_data_adjust4):
    factor_data2 = pd.DataFrame(factor_data2)
    tempaa = factor_data2.copy()
    tempaa = tempaa['weights'].unstack()
    tempbb = factor_data_adjust4.copy()
    tempbb = pd.DataFrame(tempbb)
    tempbb = tempbb[0].unstack()
    
    for i in range(tempaa.shape[0]):
        for j in range(tempaa.shape[1]):
            if pd.isnull(tempaa.iat[i,j])==True:
                tempbb.iat[i,j]=tempaa.iat[i,j]
    
    tempbb.index = pd.to_datetime(tempbb.index)
    tempbb = tempbb.stack()
    tempbb.index = tempbb.index.rename(['date','asset'])
    tempaa = factor_data2.copy()
    tempaa['weights'] = tempbb
    factor_data_new2 = tempaa.copy()    
    return factor_data_new2


def longonly_adjust_weights(group): 
    return group / group.sum()   

#%%

