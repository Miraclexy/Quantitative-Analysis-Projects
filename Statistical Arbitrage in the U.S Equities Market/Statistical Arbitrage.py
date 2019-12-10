#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 22:44:37 2019

@author: yi
"""

from zipline.api import *  
from zipline.pipeline import Pipeline
#from quantopian.research import run_pipeline
from zipline.pipeline.engine import PipelineEngine
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters.morningstar import Q500US
# sector data is not open source, should use quantopian platform to retrieve
from quantopian.pipeline.data.morningstar import Fundamentals
# not open source
from quantopian.pipeline.classifiers.morningstar import Sector
from zipline.pipeline.filters import StaticAssets
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


# get S&P500 components data
def make_pipeline():
    stock_pool = Q500US()
    yesterday_close = USEquityPricing.close.latest 
    sector = Sector(mask=stock_pool)
    pipe = Pipeline(  
        screen = stock_pool,  
        columns = {'close': yesterday_close,'sector':sector})  
    return pipe 

data = run_pipeline(make_pipeline(),'2008-01-01','2019-11-01')

data.index = data.index.rename(['date','equity'])
data['stock'] = [str(i) for i in data.index.get_level_values('equity')]

'''sector code
MORNINGSTAR_SECTOR_CODES = {
-1: 'Misc',
101: 'Basic Materials',
102: 'Consumer Cyclical',
103: 'Financial Services',
104: 'Real Estate',
205: 'Consumer Defensive',
206: 'Healthcare',
207: 'Utilities',
308: 'Communication Services',
309: 'Energy',
310: 'Industrials',
311: 'Technology' ,
}
'''

# get S&P 500 ETF data
etfs = (StaticAssets(symbols([
'XLB','XLY','XLF','IYR','XLP','XLV','XLU','IYZ','XLE','XLI','XLK' 
# Basic Materials: XLB  
# Consumer Cyclical: XLY  
# Financial Services: XLF  
# Real Estate: IYR  
# Consumer Defensive: XLP  
# Healthcare: XLV  
# Utilities: XLU  
# Communication Services: IYZ  
# Energy: XLE  
# Industrials: XLI  
# Technology: XLK  
])))
close = USEquityPricing.close.latest
pipe = Pipeline(
            columns = {'close' : close,},
            screen = etfs)

ETF = run_pipeline(pipe, '2008-01-01', '2019-11-01')
ETF.index = ETF.index.rename(['date','equity'])

# map the sector of ETF with the sector of S&P500 components
mapping = {'XLB':101,'XLE':309,'XLF':103,'XLI':310,'XLK':311,'XLP':205,'XLU':207,'XLV':206, \
           'XLY':102,'IYZ':208,'IYR':104}

ETF['equity'] = [str(i)[14:17] for i in ETF.index.get_level_values('equity')]
ETF['sector'] = ETF['equity'].map(mapping)

ETF.index = pd.MultiIndex.droplevel(ETF.index,level=1) 
data.index = pd.MultiIndex.droplevel(data.index,level=1) 

# use copy of data later
newdata = data.copy()
newetf = ETF.copy()

#%%

# Basic Materials: XLB  
XLB = newdata[['close','stock']][newdata['sector']==101]
etf_xlb = newetf[['close']][newetf['sector']==101]
XLB = XLB.set_index([XLB.index,XLB.stock])
XLB = XLB.unstack().dropna(axis=1,how='any').stack()
XLB = XLB[['close']]
delta = XLB.unstack().pct_change().shift(-2)
XLB['returns'] = delta.stack()

etf_xlb['returns'] = etf_xlb['close'].pct_change().shift(-2)

temp_xlb = XLB[['returns']].unstack()
temp_xlb['etf'] = etf_xlb.returns


# OLS regression
def regression(x,y):
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    res = model.fit()
    return res.params[1]  # beta


temp_xlb = temp_xlb.dropna(axis = 0, how='any')
betas_xlb = pd.DataFrame(index=temp_xlb.index,columns=temp_xlb.columns)

grouper = [temp_xlb.index.year]
betas_xlb['date'] = betas_xlb.index
for stock in temp_xlb.columns:
    beta = temp_xlb.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xlb[stock] = betas_xlb['date'].apply(lambda x:beta[x.year-1])

XLB['beta'] = betas_xlb.stack()['returns']
XLB = XLB.dropna(axis=0,how='any')

res_xlb = XLB.copy()
res_xlb['etf'] = np.NaN
res_xlb = res_xlb.unstack()
res_xlb['etf'] = etf_xlb['returns']
res_xlb = res_xlb.stack()

res_xlb['residual'] = res_xlb.returns-res_xlb.beta*res_xlb.etf


#==============================================================================


# test data in order to get parameters in O-U process
# test data choose one year data
test_xlb = res_xlb[['residual']][res_xlb.index.get_level_values('date')<'2010-01-01']
test_xlb = test_xlb.unstack()
test_xlb.columns = pd.MultiIndex.droplevel(test_xlb.columns,level=0)


# O-U Process in discrete time will become an AR(1) model
def ARmodel(x):
    armodel = ARIMA(x,(1,1,0))
    armodel_fit = armodel.fit()
    return armodel_fit.params


ar_xlb = pd.DataFrame(index=test_xlb.columns,columns=['a','b'])

for stock in test_xlb.columns:
    a = ARmodel(test_xlb.loc[:,stock])[0]
    b = ARmodel(test_xlb.loc[:,stock])[1]
    ar_xlb.loc[stock,'a'] = a
    ar_xlb.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xlb = pd.DataFrame(index=test_xlb.index,columns=test_xlb.columns)
for stock in test_xlb.columns:
    armodel = ARIMA(test_xlb.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xlb[stock] = armodel_fit.predict()
    
    
ar_xlb['varepsilon'] = np.var(predict_xlb-test_xlb)

res_xlb2 = res_xlb.copy()

mapping_a = ar_xlb[['a']].set_index(ar_xlb.index).T.to_dict('list')
mapping_b = ar_xlb[['b']].set_index(ar_xlb.index).T.to_dict('list')
mapping_varepsilon = ar_xlb[['varepsilon']].set_index(ar_xlb.index).T.to_dict('list')


res_xlb2['stock'] = res_xlb2.index.get_level_values('stock')
res_xlb2['a'] = res_xlb2['stock'].map(mapping_a)
res_xlb2['b'] = res_xlb2['stock'].map(mapping_b)
res_xlb2['varepsilon'] = res_xlb2['stock'].map(mapping_varepsilon)


res_xlb2['a'] = res_xlb2['a'].apply(lambda x: x[0])
res_xlb2['b'] = res_xlb2['b'].apply(lambda x: x[0])
res_xlb2['varepsilon'] = res_xlb2['varepsilon'].apply(lambda x: x[0])


res_xlb2['m'] = res_xlb2.a/(1-res_xlb2.b)
res_xlb2['sigma'] = np.sqrt(res_xlb2.varepsilon/(1-res_xlb2.b**2))


# calculate s_score
res_xlb['s_score'] = (res_xlb2.residual-res_xlb2.m)/res_xlb2.sigma
res_xlb = res_xlb[res_xlb.index.get_level_values('date')>='2010-01-01']


#%%
# define strategy, find time to change position based on s_score
def strategy(data, S_bo, S_so, S_bc, S_sc):
    '''
    parameters
        S_bo: buy to open if s_score < -S_bo
        S_so: sell to open if s_score > S_so
        S_bc: close short position if s_score < S_bc
        S_sc: close long position if s_score > S_sc
    '''
    datas = data.copy()
     
    datas['position'] = 0 # record position
    datas['flag'] = 0     # record transactions
     
    pricein = []
    priceout = []
    price_in = 1
    beta = 0
    for i in range(datas.shape[0]-1):
        # if position = 0 now, and s_score < -S_bo, then long beta times stock
        if (datas.s_score[i] < -S_bo) & (datas.position[i] == 0):
                 
            datas.loc[i,'flag'] = datas.loc[i,'beta']
            datas.loc[i+1,'position'] = 1
            date_in = datas.date[i]
            price_in = datas.close[i]
            pricein.append([date_in,price_in])
            beta = datas.loc[i,'beta']
        # if position = 0 now, and s_score > S_so, then short beta times stock
        elif (datas.s_score[i] > S_so) & (datas.position[i] == 0):
                 
            datas.loc[i,'flag'] = -datas.loc[i,'beta']
            datas.loc[i+1,'position'] = 1
            date_in = datas.date[i]
            price_in = datas.close[i]
            pricein.append([date_in,price_in])
            beta = -datas.loc[i,'beta']
        # if position = 1 now, and s_score < S_bc, then close short position
        elif (datas.position[i] == 1) & (datas.s_score[i] < S_bc):
             
            datas.loc[i,'flag'] = -beta
            datas.loc[i+1,'position'] = 0
            date_out = datas.date[i]
            price_out = datas.close[i]
            priceout.append([date_out,price_out])
        # if position = 1 now, and s_score > S_sc, then close long position
        elif (datas.position[i] == 1) & (datas.s_score[i] > S_sc):
              
            datas.loc[i,'flag'] = -beta
            datas.loc[i+1,'position'] = 0
            date_out = datas.date[i]
            price_out = datas.close[i]
            priceout.append([date_out,price_out])
         # otherwise, keep current position
        else:
            datas.loc[i+1,'position'] = datas.loc[i,'position']
    # integrate buy information and sell information  
    buyinfo = pd.DataFrame(pricein,columns=['buydate','buyprice'])
    sellinfo = pd.DataFrame(priceout,columns=['selldate','sellprice'])
    # integrate transaction information
    transaction = pd.concat([buyinfo,sellinfo], axis=1)
     
    datas = datas.reset_index(drop=True)
    datas['returns'] = datas.close.pct_change(1).shift(-1).fillna(0)
    datas['cumulative_ret'] = (1+datas.returns * datas.position).cumprod()
    datas['benchmark'] = (1+datas.returns * np.mean(datas.beta)).cumprod()
    datas['benchmark2'] = (1+datas.returns * np.std(datas.returns * datas.position) \
         /np.std(datas.returns)).cumprod()
    
    
    return datas, transaction
  
#%%
    
cumulativeret_xlb = pd.DataFrame(index = res_xlb.unstack().index, \
                                 columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_xlb['cum_ret'] = 0
cumulativeret_xlb['benchmark'] = 0
cumulativeret_xlb['benchmark2'] = 0

for stock in set(res_xlb.index.get_level_values('stock')):
    cur = res_xlb[['close','beta','s_score']][res_xlb.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)

    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xlb['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xlb['cum_ret'] += cumulativeret_xlb['temp1']
    cumulativeret_xlb['temp2'] = strategyres['benchmark']
    cumulativeret_xlb['benchmark'] += cumulativeret_xlb['temp2'] 
    cumulativeret_xlb['temp3'] = strategyres['benchmark2']
    cumulativeret_xlb['benchmark2'] += cumulativeret_xlb['temp3']
    
cumulativeret_xlb['cum_ret'] /= len(set(res_xlb.index.get_level_values('stock')))
cumulativeret_xlb['benchmark'] /= len(set(res_xlb.index.get_level_values('stock')))
cumulativeret_xlb['benchmark2'] /= len(set(res_xlb.index.get_level_values('stock')))


etf_xlb2 = etf_xlb.copy()[etf_xlb.index>="2010-01-01"]
etf_xlb2['cum_ret'] = (1+etf_xlb2.returns*np.mean(res_xlb.beta)).cumprod()

plt.plot(cumulativeret_xlb['cum_ret'],label="cumulative returns")
plt.plot(cumulativeret_xlb['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xlb['benchmark2'],label="benchmark returns2")
plt.plot(etf_xlb2['cum_ret'],label="ETF returns")
plt.legend(loc='upper left',fontsize=15)
plt.title('Basic Materials: XLB',fontsize=20)

#%%
# Consumer Cyclical: XLY 
XLY = newdata[['close','stock']][newdata['sector']==102]
etf_xly = newetf[['close']][newetf['sector']==102]
XLY = XLY.set_index([XLY.index,XLY.stock])
XLY = XLY.unstack().dropna(axis=1,how='any').stack()
XLY = XLY[['close']]
delta = XLY.unstack().pct_change().shift(-2)
XLY['returns'] = delta.stack()

etf_xly['returns'] = etf_xly['close'].pct_change().shift(-2)

temp_xly = XLY[['returns']].unstack()
temp_xly['etf'] = etf_xly.returns


temp_xly = temp_xly.dropna(axis = 0, how='any')
betas_xly = pd.DataFrame(index=temp_xly.index,columns=temp_xly.columns)

grouper = [temp_xly.index.year]
betas_xly['date'] = betas_xly.index
for stock in temp_xly.columns:
    beta = temp_xly.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xly[stock] = betas_xly['date'].apply(lambda x:beta[x.year-1])

XLY['beta'] = betas_xly.stack()['returns']
XLY = XLY.dropna(axis=0,how='any')

res_xly = XLY.copy()
res_xly['etf'] = np.NaN
res_xly = res_xly.unstack()
res_xly['etf'] = etf_xly['returns']
res_xly = res_xly.stack()

res_xly['residual'] = res_xly.returns-res_xly.beta*res_xly.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_xly = res_xly[['residual']][res_xly.index.get_level_values('date')<'2010-01-01']
test_xly = test_xly.unstack()
test_xly.columns = pd.MultiIndex.droplevel(test_xly.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_xly = pd.DataFrame(index=test_xly.columns,columns=['a','b'])

for stock in test_xly.columns:
    a = ARmodel(test_xly.loc[:,stock])[0]
    b = ARmodel(test_xly.loc[:,stock])[1]
    ar_xly.loc[stock,'a'] = a
    ar_xly.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xly = pd.DataFrame(index=test_xly.index,columns=test_xly.columns)
for stock in test_xly.columns:
    armodel = ARIMA(test_xly.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xly[stock] = armodel_fit.predict()
    
    
ar_xly['varepsilon'] = np.var(predict_xly-test_xly)

res_xly2 = res_xly.copy()

mapping_a = ar_xly[['a']].set_index(ar_xly.index).T.to_dict('list')
mapping_b = ar_xly[['b']].set_index(ar_xly.index).T.to_dict('list')
mapping_varepsilon = ar_xly[['varepsilon']].set_index(ar_xly.index).T.to_dict('list')


res_xly2['stock'] = res_xly2.index.get_level_values('stock')
res_xly2['a'] = res_xly2['stock'].map(mapping_a)
res_xly2['b'] = res_xly2['stock'].map(mapping_b)
res_xly2['varepsilon'] = res_xly2['stock'].map(mapping_varepsilon)


res_xly2['a'] = res_xly2['a'].apply(lambda x: x[0])
res_xly2['b'] = res_xly2['b'].apply(lambda x: x[0])
res_xly2['varepsilon'] = res_xly2['varepsilon'].apply(lambda x: x[0])


res_xly2['m'] = res_xly2.a/(1-res_xly2.b)
res_xly2['sigma'] = np.sqrt(res_xly2.varepsilon/(1-res_xly2.b**2))


# calculate s_score
res_xly['s_score'] = (res_xly2.residual-res_xly2.m)/res_xly2.sigma
res_xly = res_xly[res_xly.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_xly = pd.DataFrame(index = res_xly.unstack().index,columns= \
                                 ['cum_ret','benchmark','benchmark2'])
cumulativeret_xly['cum_ret'] = 0
cumulativeret_xly['benchmark'] = 0
cumulativeret_xly['benchmark2'] = 0

for stock in set(res_xly.index.get_level_values('stock')):
    cur = res_xly[['close','beta','s_score']][res_xly.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xly['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xly['cum_ret'] += cumulativeret_xly['temp1']
    cumulativeret_xly['temp2'] = strategyres['benchmark']
    cumulativeret_xly['benchmark'] += cumulativeret_xly['temp2']
    cumulativeret_xly['temp3'] = strategyres['benchmark2']
    cumulativeret_xly['benchmark2'] += cumulativeret_xly['temp3']
            
cumulativeret_xly['cum_ret'] /= len(set(res_xly.index.get_level_values('stock')))
cumulativeret_xly['benchmark'] /= len(set(res_xly.index.get_level_values('stock')))
cumulativeret_xly['benchmark2'] /= len(set(res_xly.index.get_level_values('stock')))

etf_xly2 = etf_xly.copy()[etf_xly.index>="2010-01-01"]
etf_xly2['cum_ret'] = (1+etf_xly2.returns*np.mean(res_xly.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_xly['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_xly['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xly['benchmark2'],label="benchmark returns2")
plt.plot(etf_xly2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Consumer Cyclical: XLY ',fontsize=20)

#%%
# Financial Services: XLF
XLF = newdata[['close','stock']][newdata['sector']==103]
etf_xlf = newetf[['close']][newetf['sector']==103]
XLF = XLF.set_index([XLF.index,XLF.stock])
XLF = XLF.unstack().dropna(axis=1,how='any').stack()
XLF = XLF[['close']]
delta = XLF.unstack().pct_change().shift(-2)
XLF['returns'] = delta.stack()

etf_xlf['returns'] = etf_xlf['close'].pct_change().shift(-2)

temp_xlf = XLF[['returns']].unstack()
temp_xlf['etf'] = etf_xlf.returns


temp_xlf = temp_xlf.dropna(axis = 0, how='any')
betas_xlf = pd.DataFrame(index=temp_xlf.index,columns=temp_xlf.columns)

grouper = [temp_xlf.index.year]
betas_xlf['date'] = betas_xlf.index
for stock in temp_xlf.columns:
    beta = temp_xlf.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xlf[stock] = betas_xlf['date'].apply(lambda x:beta[x.year-1])

XLF['beta'] = betas_xlf.stack()['returns']
XLF = XLF.dropna(axis=0,how='any')

res_xlf = XLF.copy()
res_xlf['etf'] = np.NaN
res_xlf = res_xlf.unstack()
res_xlf['etf'] = etf_xlf['returns']
res_xlf = res_xlf.stack()

res_xlf['residual'] = res_xlf.returns-res_xlf.beta*res_xlf.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_xlf = res_xlf[['residual']][res_xlf.index.get_level_values('date')<'2010-01-01']
test_xlf = test_xlf.unstack()
test_xlf.columns = pd.MultiIndex.droplevel(test_xlf.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_xlf = pd.DataFrame(index=test_xlf.columns,columns=['a','b'])

for stock in test_xlf.columns:
    a = ARmodel(test_xlf.loc[:,stock])[0]
    b = ARmodel(test_xlf.loc[:,stock])[1]
    ar_xlf.loc[stock,'a'] = a
    ar_xlf.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xlf = pd.DataFrame(index=test_xlf.index,columns=test_xlf.columns)
for stock in test_xlf.columns:
    armodel = ARIMA(test_xlf.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xlf[stock] = armodel_fit.predict()
    
    
ar_xlf['varepsilon'] = np.var(predict_xlf-test_xlf)

res_xlf2 = res_xlf.copy()

mapping_a = ar_xlf[['a']].set_index(ar_xlf.index).T.to_dict('list')
mapping_b = ar_xlf[['b']].set_index(ar_xlf.index).T.to_dict('list')
mapping_varepsilon = ar_xlf[['varepsilon']].set_index(ar_xlf.index).T.to_dict('list')


res_xlf2['stock'] = res_xlf2.index.get_level_values('stock')
res_xlf2['a'] = res_xlf2['stock'].map(mapping_a)
res_xlf2['b'] = res_xlf2['stock'].map(mapping_b)
res_xlf2['varepsilon'] = res_xlf2['stock'].map(mapping_varepsilon)


res_xlf2['a'] = res_xlf2['a'].apply(lambda x: x[0])
res_xlf2['b'] = res_xlf2['b'].apply(lambda x: x[0])
res_xlf2['varepsilon'] = res_xlf2['varepsilon'].apply(lambda x: x[0])


res_xlf2['m'] = res_xlf2.a/(1-res_xlf2.b)
res_xlf2['sigma'] = np.sqrt(res_xlf2.varepsilon/(1-res_xlf2.b**2))


# calculate s_score
res_xlf['s_score'] = (res_xlf2.residual-res_xlf2.m)/res_xlf2.sigma
res_xlf = res_xlf[res_xlf.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_xlf = pd.DataFrame(index = res_xlf.unstack().index,columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_xlf['cum_ret'] = 0
cumulativeret_xlf['benchmark'] = 0
cumulativeret_xlf['benchmark2'] = 0

for stock in set(res_xlf.index.get_level_values('stock')):
    cur = res_xlf[['close','beta','s_score']][res_xlf.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xlf['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xlf['cum_ret'] += cumulativeret_xlf['temp1']
    cumulativeret_xlf['temp2'] = strategyres['benchmark']
    cumulativeret_xlf['benchmark'] += cumulativeret_xlf['temp2']
    cumulativeret_xlf['temp3'] = strategyres['benchmark2']
    cumulativeret_xlf['benchmark2'] += cumulativeret_xlf['temp3']
            
cumulativeret_xlf['cum_ret'] /= len(set(res_xlf.index.get_level_values('stock')))
cumulativeret_xlf['benchmark'] /= len(set(res_xlf.index.get_level_values('stock')))
cumulativeret_xlf['benchmark2'] /= len(set(res_xlf.index.get_level_values('stock')))

etf_xlf2 = etf_xlf.copy()[etf_xlf.index>="2010-01-01"]
etf_xlf2['cum_ret'] = (1+etf_xlf2.returns*np.mean(res_xlf.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_xlf['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_xlf['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xlf['benchmark2'],label="benchmark returns2")
plt.plot(etf_xlf2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Financial Services: XLF',fontsize=20)

#%%
# Real Estate: IYR
IYR = newdata[['close','stock']][newdata['sector']==104]
etf_iyr = newetf[['close']][newetf['sector']==104]
IYR = IYR.set_index([IYR.index,IYR.stock])
IYR = IYR.unstack().dropna(axis=1,how='any').stack()
IYR = IYR[['close']]
delta = IYR.unstack().pct_change().shift(-2)
IYR['returns'] = delta.stack()

etf_iyr['returns'] = etf_iyr['close'].pct_change().shift(-2)

temp_iyr = IYR[['returns']].unstack()
temp_iyr['etf'] = etf_iyr.returns


temp_iyr = temp_iyr.dropna(axis = 0, how='any')
betas_iyr = pd.DataFrame(index=temp_iyr.index,columns=temp_iyr.columns)

grouper = [temp_iyr.index.year]
betas_iyr['date'] = betas_iyr.index
for stock in temp_iyr.columns:
    beta = temp_iyr.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_iyr[stock] = betas_iyr['date'].apply(lambda x:beta[x.year-1])

IYR['beta'] = betas_iyr.stack()['returns']
IYR = IYR.dropna(axis=0,how='any')

res_iyr = IYR.copy()
res_iyr['etf'] = np.NaN
res_iyr = res_iyr.unstack()
res_iyr['etf'] = etf_iyr['returns']
res_iyr = res_iyr.stack()

res_iyr['residual'] = res_iyr.returns-res_iyr.beta*res_iyr.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_iyr = res_iyr[['residual']][res_iyr.index.get_level_values('date')<'2010-01-01']
test_iyr = test_iyr.unstack()
test_iyr.columns = pd.MultiIndex.droplevel(test_iyr.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_iyr = pd.DataFrame(index=test_iyr.columns,columns=['a','b'])

for stock in test_iyr.columns:
    a = ARmodel(test_iyr.loc[:,stock])[0]
    b = ARmodel(test_iyr.loc[:,stock])[1]
    ar_iyr.loc[stock,'a'] = a
    ar_iyr.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_iyr = pd.DataFrame(index=test_iyr.index,columns=test_iyr.columns)
for stock in test_iyr.columns:
    armodel = ARIMA(test_iyr.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_iyr[stock] = armodel_fit.predict()
    
    
ar_iyr['varepsilon'] = np.var(predict_iyr-test_iyr)

res_iyr2 = res_iyr.copy()

mapping_a = ar_iyr[['a']].set_index(ar_iyr.index).T.to_dict('list')
mapping_b = ar_iyr[['b']].set_index(ar_iyr.index).T.to_dict('list')
mapping_varepsilon = ar_iyr[['varepsilon']].set_index(ar_iyr.index).T.to_dict('list')


res_iyr2['stock'] = res_iyr2.index.get_level_values('stock')
res_iyr2['a'] = res_iyr2['stock'].map(mapping_a)
res_iyr2['b'] = res_iyr2['stock'].map(mapping_b)
res_iyr2['varepsilon'] = res_iyr2['stock'].map(mapping_varepsilon)


res_iyr2['a'] = res_iyr2['a'].apply(lambda x: x[0])
res_iyr2['b'] = res_iyr2['b'].apply(lambda x: x[0])
res_iyr2['varepsilon'] = res_iyr2['varepsilon'].apply(lambda x: x[0])


res_iyr2['m'] = res_iyr2.a/(1-res_iyr2.b)
res_iyr2['sigma'] = np.sqrt(res_iyr2.varepsilon/(1-res_iyr2.b**2))


# calculate s_score
res_iyr['s_score'] = (res_iyr2.residual-res_iyr2.m)/res_iyr2.sigma
res_iyr = res_iyr[res_iyr.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_iyr = pd.DataFrame(index = res_iyr.unstack().index,columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_iyr['cum_ret'] = 0
cumulativeret_iyr['benchmark'] = 0
cumulativeret_iyr['benchmark2'] = 0

for stock in set(res_iyr.index.get_level_values('stock')):
    cur = res_iyr[['close','beta','s_score']][res_iyr.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_iyr['temp1'] = strategyres['cumulative_ret']
    cumulativeret_iyr['cum_ret'] += cumulativeret_iyr['temp1']
    cumulativeret_iyr['temp2'] = strategyres['benchmark']
    cumulativeret_iyr['benchmark'] += cumulativeret_iyr['temp2']
    cumulativeret_iyr['temp3'] = strategyres['benchmark2']
    cumulativeret_iyr['benchmark2'] += cumulativeret_iyr['temp3']
            
cumulativeret_iyr['cum_ret'] /= len(set(res_iyr.index.get_level_values('stock')))
cumulativeret_iyr['benchmark'] /= len(set(res_iyr.index.get_level_values('stock')))
cumulativeret_iyr['benchmark2'] /= len(set(res_iyr.index.get_level_values('stock')))

etf_iyr2 = etf_iyr.copy()[etf_iyr.index>="2010-01-01"]
etf_iyr2['cum_ret'] = (1+etf_iyr2.returns*np.mean(res_iyr.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_iyr['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_iyr['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_iyr['benchmark2'],label="benchmark returns2")
plt.plot(etf_iyr2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Real Estate: IYR',fontsize=20)

#%%
# Consumer Defensive: XLP  
XLP = newdata[['close','stock']][newdata['sector']==205]
etf_xlp = newetf[['close']][newetf['sector']==205]
XLP = XLP.set_index([XLP.index,XLP.stock])
XLP = XLP.unstack().dropna(axis=1,how='any').stack()
XLP = XLP[['close']]
delta = XLP.unstack().pct_change().shift(-2)
XLP['returns'] = delta.stack()

etf_xlp['returns'] = etf_xlp['close'].pct_change().shift(-2)

temp_xlp = XLP[['returns']].unstack()
temp_xlp['etf'] = etf_xlp.returns


temp_xlp = temp_xlp.dropna(axis = 0, how='any')
betas_xlp = pd.DataFrame(index=temp_xlp.index,columns=temp_xlp.columns)

grouper = [temp_xlp.index.year]
betas_xlp['date'] = betas_xlp.index
for stock in temp_xlp.columns:
    beta = temp_xlp.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xlp[stock] = betas_xlp['date'].apply(lambda x:beta[x.year-1])

XLP['beta'] = betas_xlp.stack()['returns']
XLP = XLP.dropna(axis=0,how='any')

res_xlp = XLP.copy()
res_xlp['etf'] = np.NaN
res_xlp = res_xlp.unstack()
res_xlp['etf'] = etf_xlp['returns']
res_xlp = res_xlp.stack()

res_xlp['residual'] = res_xlp.returns-res_xlp.beta*res_xlp.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_xlp = res_xlp[['residual']][res_xlp.index.get_level_values('date')<'2010-01-01']
test_xlp = test_xlp.unstack()
test_xlp.columns = pd.MultiIndex.droplevel(test_xlp.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_xlp = pd.DataFrame(index=test_xlp.columns,columns=['a','b'])

for stock in test_xlp.columns:
    a = ARmodel(test_xlp.loc[:,stock])[0]
    b = ARmodel(test_xlp.loc[:,stock])[1]
    ar_xlp.loc[stock,'a'] = a
    ar_xlp.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xlp = pd.DataFrame(index=test_xlp.index,columns=test_xlp.columns)
for stock in test_xlp.columns:
    armodel = ARIMA(test_xlp.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xlp[stock] = armodel_fit.predict()
    
    
ar_xlp['varepsilon'] = np.var(predict_xlp-test_xlp)

res_xlp2 = res_xlp.copy()

mapping_a = ar_xlp[['a']].set_index(ar_xlp.index).T.to_dict('list')
mapping_b = ar_xlp[['b']].set_index(ar_xlp.index).T.to_dict('list')
mapping_varepsilon = ar_xlp[['varepsilon']].set_index(ar_xlp.index).T.to_dict('list')


res_xlp2['stock'] = res_xlp2.index.get_level_values('stock')
res_xlp2['a'] = res_xlp2['stock'].map(mapping_a)
res_xlp2['b'] = res_xlp2['stock'].map(mapping_b)
res_xlp2['varepsilon'] = res_xlp2['stock'].map(mapping_varepsilon)


res_xlp2['a'] = res_xlp2['a'].apply(lambda x: x[0])
res_xlp2['b'] = res_xlp2['b'].apply(lambda x: x[0])
res_xlp2['varepsilon'] = res_xlp2['varepsilon'].apply(lambda x: x[0])


res_xlp2['m'] = res_xlp2.a/(1-res_xlp2.b)
res_xlp2['sigma'] = np.sqrt(res_xlp2.varepsilon/(1-res_xlp2.b**2))


# calculate s_score
res_xlp['s_score'] = (res_xlp2.residual-res_xlp2.m)/res_xlp2.sigma
res_xlp = res_xlp[res_xlp.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_xlp = pd.DataFrame(index = res_xlp.unstack().index,columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_xlp['cum_ret'] = 0
cumulativeret_xlp['benchmark'] = 0
cumulativeret_xlp['benchmark2'] = 0


for stock in set(res_xlp.index.get_level_values('stock')):
    cur = res_xlp[['close','beta','s_score']][res_xlp.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xlp['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xlp['cum_ret'] += cumulativeret_xlp['temp1']
    cumulativeret_xlp['temp2'] = strategyres['benchmark']
    cumulativeret_xlp['benchmark'] += cumulativeret_xlp['temp2']
    cumulativeret_xlp['temp3'] = strategyres['benchmark2']
    cumulativeret_xlp['benchmark2'] += cumulativeret_xlp['temp3']
            
cumulativeret_xlp['cum_ret'] /= len(set(res_xlp.index.get_level_values('stock')))
cumulativeret_xlp['benchmark'] /= len(set(res_xlp.index.get_level_values('stock')))
cumulativeret_xlp['benchmark2'] /= len(set(res_xlp.index.get_level_values('stock')))

etf_xlp2 = etf_xlp.copy()[etf_xlp.index>="2010-01-01"]
etf_xlp2['cum_ret'] = (1+etf_xlp2.returns*np.mean(res_xlp.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_xlp['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_xlp['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xlp['benchmark2'],label="benchmark returns2")
plt.plot(etf_xlp2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Consumer Defensive: XLP',fontsize=20) 

#%%
# Healthcare: XLV  
XLV = newdata[['close','stock']][newdata['sector']==206]
etf_xlv = newetf[['close']][newetf['sector']==206]
XLV = XLV.set_index([XLV.index,XLV.stock])
XLV = XLV.unstack().dropna(axis=1,how='any').stack()
XLV = XLV[['close']]
delta = XLV.unstack().pct_change().shift(-2)
XLV['returns'] = delta.stack()

etf_xlv['returns'] = etf_xlv['close'].pct_change().shift(-2)

temp_xlv = XLV[['returns']].unstack()
temp_xlv['etf'] = etf_xlv.returns


temp_xlv = temp_xlv.dropna(axis = 0, how='any')
betas_xlv = pd.DataFrame(index=temp_xlv.index,columns=temp_xlv.columns)

grouper = [temp_xlv.index.year]
betas_xlv['date'] = betas_xlv.index
for stock in temp_xlv.columns:
    beta = temp_xlv.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xlv[stock] = betas_xlv['date'].apply(lambda x:beta[x.year-1])

XLV['beta'] = betas_xlv.stack()['returns']
XLV = XLV.dropna(axis=0,how='any')

res_xlv = XLV.copy()
res_xlv['etf'] = np.NaN
res_xlv = res_xlv.unstack()
res_xlv['etf'] = etf_xlv['returns']
res_xlv = res_xlv.stack()

res_xlv['residual'] = res_xlv.returns-res_xlv.beta*res_xlv.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_xlv = res_xlv[['residual']][res_xlv.index.get_level_values('date')<'2010-01-01']
test_xlv = test_xlv.unstack()
test_xlv.columns = pd.MultiIndex.droplevel(test_xlv.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_xlv = pd.DataFrame(index=test_xlv.columns,columns=['a','b'])

for stock in test_xlv.columns:
    a = ARmodel(test_xlv.loc[:,stock])[0]
    b = ARmodel(test_xlv.loc[:,stock])[1]
    ar_xlv.loc[stock,'a'] = a
    ar_xlv.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xlv = pd.DataFrame(index=test_xlv.index,columns=test_xlv.columns)
for stock in test_xlv.columns:
    armodel = ARIMA(test_xlv.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xlv[stock] = armodel_fit.predict()
    
    
ar_xlv['varepsilon'] = np.var(predict_xlv-test_xlv)

res_xlv2 = res_xlv.copy()

mapping_a = ar_xlv[['a']].set_index(ar_xlv.index).T.to_dict('list')
mapping_b = ar_xlv[['b']].set_index(ar_xlv.index).T.to_dict('list')
mapping_varepsilon = ar_xlv[['varepsilon']].set_index(ar_xlv.index).T.to_dict('list')


res_xlv2['stock'] = res_xlv2.index.get_level_values('stock')
res_xlv2['a'] = res_xlv2['stock'].map(mapping_a)
res_xlv2['b'] = res_xlv2['stock'].map(mapping_b)
res_xlv2['varepsilon'] = res_xlv2['stock'].map(mapping_varepsilon)


res_xlv2['a'] = res_xlv2['a'].apply(lambda x: x[0])
res_xlv2['b'] = res_xlv2['b'].apply(lambda x: x[0])
res_xlv2['varepsilon'] = res_xlv2['varepsilon'].apply(lambda x: x[0])


res_xlv2['m'] = res_xlv2.a/(1-res_xlv2.b)
res_xlv2['sigma'] = np.sqrt(res_xlv2.varepsilon/(1-res_xlv2.b**2))


# calculate s_score
res_xlv['s_score'] = (res_xlv2.residual-res_xlv2.m)/res_xlv2.sigma
res_xlv = res_xlv[res_xlv.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_xlv = pd.DataFrame(index = res_xlv.unstack().index,columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_xlv['cum_ret'] = 0
cumulativeret_xlv['benchmark'] = 0
cumulativeret_xlv['benchmark2'] = 0

for stock in set(res_xlv.index.get_level_values('stock')):
    cur = res_xlv[['close','beta','s_score']][res_xlv.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xlv['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xlv['cum_ret'] += cumulativeret_xlv['temp1']
    cumulativeret_xlv['temp2'] = strategyres['benchmark']
    cumulativeret_xlv['benchmark'] += cumulativeret_xlv['temp2']
    cumulativeret_xlv['temp3'] = strategyres['benchmark2']
    cumulativeret_xlv['benchmark2'] += cumulativeret_xlv['temp3']
            
cumulativeret_xlv['cum_ret'] /= len(set(res_xlv.index.get_level_values('stock')))
cumulativeret_xlv['benchmark'] /= len(set(res_xlv.index.get_level_values('stock')))
cumulativeret_xlv['benchmark2'] /= len(set(res_xlv.index.get_level_values('stock')))

etf_xlv2 = etf_xlv.copy()[etf_xlv.index>="2010-01-01"]
etf_xlv2['cum_ret'] = (1+etf_xlv2.returns*np.mean(res_xlv.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_xlv['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_xlv['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xlv['benchmark2'],label="benchmark returns2")
plt.plot(etf_xlv2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Healthcare: XLV',fontsize=20)

#%%
# Utilities: XLU 
XLU = newdata[['close','stock']][newdata['sector']==207]
etf_xlu = newetf[['close']][newetf['sector']==207]
XLU = XLU.set_index([XLU.index,XLU.stock])
XLU = XLU.unstack().dropna(axis=1,how='any').stack()
XLU = XLU[['close']]
delta = XLU.unstack().pct_change().shift(-2)
XLU['returns'] = delta.stack()

etf_xlu['returns'] = etf_xlu['close'].pct_change().shift(-2)

temp_xlu = XLU[['returns']].unstack()
temp_xlu['etf'] = etf_xlu.returns


temp_xlu = temp_xlu.dropna(axis = 0, how='any')
betas_xlu = pd.DataFrame(index=temp_xlu.index,columns=temp_xlu.columns)

grouper = [temp_xlu.index.year]
betas_xlu['date'] = betas_xlu.index
for stock in temp_xlu.columns:
    beta = temp_xlu.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xlu[stock] = betas_xlu['date'].apply(lambda x:beta[x.year-1])

XLU['beta'] = betas_xlu.stack()['returns']
XLU = XLU.dropna(axis=0,how='any')

res_xlu = XLU.copy()
res_xlu['etf'] = np.NaN
res_xlu = res_xlu.unstack()
res_xlu['etf'] = etf_xlu['returns']
res_xlu = res_xlu.stack()

res_xlu['residual'] = res_xlu.returns-res_xlu.beta*res_xlu.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_xlu = res_xlu[['residual']][res_xlu.index.get_level_values('date')<'2010-01-01']
test_xlu = test_xlu.unstack()
test_xlu.columns = pd.MultiIndex.droplevel(test_xlu.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_xlu = pd.DataFrame(index=test_xlu.columns,columns=['a','b'])

for stock in test_xlu.columns:
    a = ARmodel(test_xlu.loc[:,stock])[0]
    b = ARmodel(test_xlu.loc[:,stock])[1]
    ar_xlu.loc[stock,'a'] = a
    ar_xlu.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xlu = pd.DataFrame(index=test_xlu.index,columns=test_xlu.columns)
for stock in test_xlu.columns:
    armodel = ARIMA(test_xlu.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xlu[stock] = armodel_fit.predict()
    
    
ar_xlu['varepsilon'] = np.var(predict_xlu-test_xlu)

res_xlu2 = res_xlu.copy()

mapping_a = ar_xlu[['a']].set_index(ar_xlu.index).T.to_dict('list')
mapping_b = ar_xlu[['b']].set_index(ar_xlu.index).T.to_dict('list')
mapping_varepsilon = ar_xlu[['varepsilon']].set_index(ar_xlu.index).T.to_dict('list')


res_xlu2['stock'] = res_xlu2.index.get_level_values('stock')
res_xlu2['a'] = res_xlu2['stock'].map(mapping_a)
res_xlu2['b'] = res_xlu2['stock'].map(mapping_b)
res_xlu2['varepsilon'] = res_xlu2['stock'].map(mapping_varepsilon)


res_xlu2['a'] = res_xlu2['a'].apply(lambda x: x[0])
res_xlu2['b'] = res_xlu2['b'].apply(lambda x: x[0])
res_xlu2['varepsilon'] = res_xlu2['varepsilon'].apply(lambda x: x[0])


res_xlu2['m'] = res_xlu2.a/(1-res_xlu2.b)
res_xlu2['sigma'] = np.sqrt(res_xlu2.varepsilon/(1-res_xlu2.b**2))


# calculate s_score
res_xlu['s_score'] = (res_xlu2.residual-res_xlu2.m)/res_xlu2.sigma
res_xlu = res_xlu[res_xlu.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_xlu = pd.DataFrame(index = res_xlu.unstack().index,columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_xlu['cum_ret'] = 0
cumulativeret_xlu['benchmark'] = 0
cumulativeret_xlu['benchmark2'] = 0

for stock in set(res_xlu.index.get_level_values('stock')):
    cur = res_xlu[['close','beta','s_score']][res_xlu.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xlu['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xlu['cum_ret'] += cumulativeret_xlu['temp1']
    cumulativeret_xlu['temp2'] = strategyres['benchmark']
    cumulativeret_xlu['benchmark'] += cumulativeret_xlu['temp2']
    cumulativeret_xlu['temp3'] = strategyres['benchmark2']
    cumulativeret_xlu['benchmark2'] += cumulativeret_xlu['temp3']    
            
cumulativeret_xlu['cum_ret'] /= len(set(res_xlu.index.get_level_values('stock')))
cumulativeret_xlu['benchmark'] /= len(set(res_xlu.index.get_level_values('stock')))
cumulativeret_xlu['benchmark2'] /= len(set(res_xlu.index.get_level_values('stock')))

etf_xlu2 = etf_xlu.copy()[etf_xlu.index>="2010-01-01"]
etf_xlu2['cum_ret'] = (1+etf_xlu2.returns*np.mean(res_xlu.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_xlu['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_xlu['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xlu['benchmark2'],label="benchmark returns2")
plt.plot(etf_xlu2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Utilities: XLU',fontsize=20)

#%%
# Communication Services: IYZ (do not have ETF data)
# Energy: XLE 
XLE = newdata[['close','stock']][newdata['sector']==309]
etf_xle = newetf[['close']][newetf['sector']==309]
XLE = XLE.set_index([XLE.index,XLE.stock])
XLE = XLE.unstack().dropna(axis=1,how='any').stack()
XLE = XLE[['close']]
delta = XLE.unstack().pct_change().shift(-2)
XLE['returns'] = delta.stack()

etf_xle['returns'] = etf_xle['close'].pct_change().shift(-2)

temp_xle = XLE[['returns']].unstack()
temp_xle['etf'] = etf_xle.returns


temp_xle = temp_xle.dropna(axis = 0, how='any')
betas_xle = pd.DataFrame(index=temp_xle.index,columns=temp_xle.columns)

grouper = [temp_xle.index.year]
betas_xle['date'] = betas_xle.index
for stock in temp_xle.columns:
    beta = temp_xle.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xle[stock] = betas_xle['date'].apply(lambda x:beta[x.year-1])

XLE['beta'] = betas_xle.stack()['returns']
XLE = XLE.dropna(axis=0,how='any')

res_xle = XLE.copy()
res_xle['etf'] = np.NaN
res_xle = res_xle.unstack()
res_xle['etf'] = etf_xle['returns']
res_xle = res_xle.stack()

res_xle['residual'] = res_xle.returns-res_xle.beta*res_xle.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_xle = res_xle[['residual']][res_xle.index.get_level_values('date')<'2010-01-01']
test_xle = test_xle.unstack()
test_xle.columns = pd.MultiIndex.droplevel(test_xle.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_xle = pd.DataFrame(index=test_xle.columns,columns=['a','b'])

for stock in test_xle.columns:
    a = ARmodel(test_xle.loc[:,stock])[0]
    b = ARmodel(test_xle.loc[:,stock])[1]
    ar_xle.loc[stock,'a'] = a
    ar_xle.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xle = pd.DataFrame(index=test_xle.index,columns=test_xle.columns)
for stock in test_xle.columns:
    armodel = ARIMA(test_xle.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xle[stock] = armodel_fit.predict()
    
    
ar_xle['varepsilon'] = np.var(predict_xle-test_xle)

res_xle2 = res_xle.copy()

mapping_a = ar_xle[['a']].set_index(ar_xle.index).T.to_dict('list')
mapping_b = ar_xle[['b']].set_index(ar_xle.index).T.to_dict('list')
mapping_varepsilon = ar_xle[['varepsilon']].set_index(ar_xle.index).T.to_dict('list')


res_xle2['stock'] = res_xle2.index.get_level_values('stock')
res_xle2['a'] = res_xle2['stock'].map(mapping_a)
res_xle2['b'] = res_xle2['stock'].map(mapping_b)
res_xle2['varepsilon'] = res_xle2['stock'].map(mapping_varepsilon)


res_xle2['a'] = res_xle2['a'].apply(lambda x: x[0])
res_xle2['b'] = res_xle2['b'].apply(lambda x: x[0])
res_xle2['varepsilon'] = res_xle2['varepsilon'].apply(lambda x: x[0])


res_xle2['m'] = res_xle2.a/(1-res_xle2.b)
res_xle2['sigma'] = np.sqrt(res_xle2.varepsilon/(1-res_xle2.b**2))


# calculate s_score
res_xle['s_score'] = (res_xle2.residual-res_xle2.m)/res_xle2.sigma
res_xle = res_xle[res_xle.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_xle = pd.DataFrame(index = res_xle.unstack().index,columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_xle['cum_ret'] = 0
cumulativeret_xle['benchmark'] = 0
cumulativeret_xle['benchmark2'] = 0

for stock in set(res_xle.index.get_level_values('stock')):
    cur = res_xle[['close','beta','s_score']][res_xle.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xle['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xle['cum_ret'] += cumulativeret_xle['temp1']
    cumulativeret_xle['temp2'] = strategyres['benchmark']
    cumulativeret_xle['benchmark'] += cumulativeret_xle['temp2']
    cumulativeret_xle['temp3'] = strategyres['benchmark2']
    cumulativeret_xle['benchmark2'] += cumulativeret_xle['temp3']
            
cumulativeret_xle['cum_ret'] /= len(set(res_xle.index.get_level_values('stock')))
cumulativeret_xle['benchmark'] /= len(set(res_xle.index.get_level_values('stock')))
cumulativeret_xle['benchmark2'] /= len(set(res_xle.index.get_level_values('stock')))

etf_xle2 = etf_xle.copy()[etf_xle.index>="2010-01-01"]
etf_xle2['cum_ret'] = (1+etf_xle2.returns*np.mean(res_xle.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_xle['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_xle['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xle['benchmark2'],label="benchmark returns2")
plt.plot(etf_xle2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Energy: XLE',fontsize=20)

#%%
# Industrials: XLI 
XLI = newdata[['close','stock']][newdata['sector']==310]
etf_xli = newetf[['close']][newetf['sector']==310]
XLI = XLI.set_index([XLI.index,XLI.stock])
XLI = XLI.unstack().dropna(axis=1,how='any').stack()
XLI = XLI[['close']]
delta = XLI.unstack().pct_change().shift(-2)
XLI['returns'] = delta.stack()

etf_xli['returns'] = etf_xli['close'].pct_change().shift(-2)

temp_xli = XLI[['returns']].unstack()
temp_xli['etf'] = etf_xli.returns


temp_xli = temp_xli.dropna(axis = 0, how='any')
betas_xli = pd.DataFrame(index=temp_xli.index,columns=temp_xli.columns)

grouper = [temp_xli.index.year]
betas_xli['date'] = betas_xli.index
for stock in temp_xli.columns:
    beta = temp_xli.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xli[stock] = betas_xli['date'].apply(lambda x:beta[x.year-1])

XLI['beta'] = betas_xli.stack()['returns']
XLI = XLI.dropna(axis=0,how='any')

res_xli = XLI.copy()
res_xli['etf'] = np.NaN
res_xli = res_xli.unstack()
res_xli['etf'] = etf_xli['returns']
res_xli = res_xli.stack()

res_xli['residual'] = res_xli.returns-res_xli.beta*res_xli.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_xli = res_xli[['residual']][res_xli.index.get_level_values('date')<'2010-01-01']
test_xli = test_xli.unstack()
test_xli.columns = pd.MultiIndex.droplevel(test_xli.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_xli = pd.DataFrame(index=test_xli.columns,columns=['a','b'])

for stock in test_xli.columns:
    a = ARmodel(test_xli.loc[:,stock])[0]
    b = ARmodel(test_xli.loc[:,stock])[1]
    ar_xli.loc[stock,'a'] = a
    ar_xli.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xli = pd.DataFrame(index=test_xli.index,columns=test_xli.columns)
for stock in test_xli.columns:
    armodel = ARIMA(test_xli.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xli[stock] = armodel_fit.predict()
    
    
ar_xli['varepsilon'] = np.var(predict_xli-test_xli)

res_xli2 = res_xli.copy()

mapping_a = ar_xli[['a']].set_index(ar_xli.index).T.to_dict('list')
mapping_b = ar_xli[['b']].set_index(ar_xli.index).T.to_dict('list')
mapping_varepsilon = ar_xli[['varepsilon']].set_index(ar_xli.index).T.to_dict('list')


res_xli2['stock'] = res_xli2.index.get_level_values('stock')
res_xli2['a'] = res_xli2['stock'].map(mapping_a)
res_xli2['b'] = res_xli2['stock'].map(mapping_b)
res_xli2['varepsilon'] = res_xli2['stock'].map(mapping_varepsilon)


res_xli2['a'] = res_xli2['a'].apply(lambda x: x[0])
res_xli2['b'] = res_xli2['b'].apply(lambda x: x[0])
res_xli2['varepsilon'] = res_xli2['varepsilon'].apply(lambda x: x[0])


res_xli2['m'] = res_xli2.a/(1-res_xli2.b)
res_xli2['sigma'] = np.sqrt(res_xli2.varepsilon/(1-res_xli2.b**2))


# calculate s_score
res_xli['s_score'] = (res_xli2.residual-res_xli2.m)/res_xli2.sigma
res_xli = res_xli[res_xli.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_xli = pd.DataFrame(index = res_xli.unstack().index,columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_xli['cum_ret'] = 0
cumulativeret_xli['benchmark'] = 0
cumulativeret_xli['benchmark2'] = 0

for stock in set(res_xli.index.get_level_values('stock')):
    cur = res_xli[['close','beta','s_score']][res_xli.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xli['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xli['cum_ret'] += cumulativeret_xli['temp1']
    cumulativeret_xli['temp2'] = strategyres['benchmark']
    cumulativeret_xli['benchmark'] += cumulativeret_xli['temp2']
    cumulativeret_xli['temp3'] = strategyres['benchmark2']
    cumulativeret_xli['benchmark2'] += cumulativeret_xli['temp3']    
            
cumulativeret_xli['cum_ret'] /= len(set(res_xli.index.get_level_values('stock')))
cumulativeret_xli['benchmark'] /= len(set(res_xli.index.get_level_values('stock')))
cumulativeret_xli['benchmark2'] /= len(set(res_xli.index.get_level_values('stock')))

etf_xli2 = etf_xli.copy()[etf_xli.index>="2010-01-01"]
etf_xli2['cum_ret'] = (1+etf_xli2.returns*np.mean(res_xli.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_xli['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_xli['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xli['benchmark2'],label="benchmark returns2")
plt.plot(etf_xli2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Industrials: XLI',fontsize=20)

#%%
# Technology: XLK 
XLK = newdata[['close','stock']][newdata['sector']==311]
etf_xlk = newetf[['close']][newetf['sector']==311]
XLK = XLK.set_index([XLK.index,XLK.stock])
XLK = XLK.unstack().dropna(axis=1,how='any').stack()
XLK = XLK[['close']]
delta = XLK.unstack().pct_change().shift(-2)
XLK['returns'] = delta.stack()

etf_xlk['returns'] = etf_xlk['close'].pct_change().shift(-2)

temp_xlk = XLK[['returns']].unstack()
temp_xlk['etf'] = etf_xlk.returns


temp_xlk = temp_xlk.dropna(axis = 0, how='any')
betas_xlk = pd.DataFrame(index=temp_xlk.index,columns=temp_xlk.columns)

grouper = [temp_xlk.index.year]
betas_xlk['date'] = betas_xlk.index
for stock in temp_xlk.columns:
    beta = temp_xlk.groupby(grouper).apply(lambda x:regression(x.loc[:,stock],x.iloc[:,-1]))
    beta[2007] = np.NaN
    betas_xlk[stock] = betas_xlk['date'].apply(lambda x:beta[x.year-1])

XLK['beta'] = betas_xlk.stack()['returns']
XLK = XLK.dropna(axis=0,how='any')

res_xlk = XLK.copy()
res_xlk['etf'] = np.NaN
res_xlk = res_xlk.unstack()
res_xlk['etf'] = etf_xlk['returns']
res_xlk = res_xlk.stack()

res_xlk['residual'] = res_xlk.returns-res_xlk.beta*res_xlk.etf


# test data in order to get parameters in O-U process
# test data choose one year data
test_xlk = res_xlk[['residual']][res_xlk.index.get_level_values('date')<'2010-01-01']
test_xlk = test_xlk.unstack()
test_xlk.columns = pd.MultiIndex.droplevel(test_xlk.columns,level=0)




# O-U Process in discrete time will become an AR(1) model
ar_xlk = pd.DataFrame(index=test_xlk.columns,columns=['a','b'])

for stock in test_xlk.columns:
    a = ARmodel(test_xlk.loc[:,stock])[0]
    b = ARmodel(test_xlk.loc[:,stock])[1]
    ar_xlk.loc[stock,'a'] = a
    ar_xlk.loc[stock,'b'] = b  
    
    
# calculate variance of sigma    
predict_xlk = pd.DataFrame(index=test_xlk.index,columns=test_xlk.columns)
for stock in test_xlk.columns:
    armodel = ARIMA(test_xlk.loc[:,stock],(1,1,0))
    armodel_fit = armodel.fit()
    predict_xlk[stock] = armodel_fit.predict()
    
    
ar_xlk['varepsilon'] = np.var(predict_xlk-test_xlk)

res_xlk2 = res_xlk.copy()

mapping_a = ar_xlk[['a']].set_index(ar_xlk.index).T.to_dict('list')
mapping_b = ar_xlk[['b']].set_index(ar_xlk.index).T.to_dict('list')
mapping_varepsilon = ar_xlk[['varepsilon']].set_index(ar_xlk.index).T.to_dict('list')


res_xlk2['stock'] = res_xlk2.index.get_level_values('stock')
res_xlk2['a'] = res_xlk2['stock'].map(mapping_a)
res_xlk2['b'] = res_xlk2['stock'].map(mapping_b)
res_xlk2['varepsilon'] = res_xlk2['stock'].map(mapping_varepsilon)


res_xlk2['a'] = res_xlk2['a'].apply(lambda x: x[0])
res_xlk2['b'] = res_xlk2['b'].apply(lambda x: x[0])
res_xlk2['varepsilon'] = res_xlk2['varepsilon'].apply(lambda x: x[0])


res_xlk2['m'] = res_xlk2.a/(1-res_xlk2.b)
res_xlk2['sigma'] = np.sqrt(res_xlk2.varepsilon/(1-res_xlk2.b**2))


# calculate s_score
res_xlk['s_score'] = (res_xlk2.residual-res_xlk2.m)/res_xlk2.sigma
res_xlk = res_xlk[res_xlk.index.get_level_values('date')>='2010-01-01']



# begin strategy
cumulativeret_xlk = pd.DataFrame(index = res_xlk.unstack().index,columns=['cum_ret','benchmark','benchmark2'])
cumulativeret_xlk['cum_ret'] = 0
cumulativeret_xlk['benchmark'] = 0
cumulativeret_xlk['benchmark2'] = 0

for stock in set(res_xlk.index.get_level_values('stock')):
    cur = res_xlk[['close','beta','s_score']][res_xlk.index.get_level_values('stock')==stock]
    cur.index = pd.MultiIndex.droplevel(cur.index,level=1)
    cur = cur.reset_index()

    strategyres, transaction = strategy(cur,0.1,0.1,-0.15,-0.2)
#    finalresult, yearlyresult = performance(transaction,strategyres)
    strategyres.set_index(strategyres['date'],inplace=True)
    cumulativeret_xlk['temp1'] = strategyres['cumulative_ret']
    cumulativeret_xlk['cum_ret'] += cumulativeret_xlk['temp1']
    cumulativeret_xlk['temp2'] = strategyres['benchmark']
    cumulativeret_xlk['benchmark'] += cumulativeret_xlk['temp2']
    cumulativeret_xlk['temp3'] = strategyres['benchmark2']
    cumulativeret_xlk['benchmark2'] += cumulativeret_xlk['temp3']  
            
cumulativeret_xlk['cum_ret'] /= len(set(res_xlk.index.get_level_values('stock')))
cumulativeret_xlk['benchmark'] /= len(set(res_xlk.index.get_level_values('stock')))
cumulativeret_xlk['benchmark2'] /= len(set(res_xlk.index.get_level_values('stock')))

etf_xlk2 = etf_xlk.copy()[etf_xlk.index>="2010-01-01"]
etf_xlk2['cum_ret'] = (1+etf_xlk2.returns*np.mean(res_xlk.beta)).cumprod()
# plot cumulative returns
plt.plot(cumulativeret_xlk['cum_ret'],label="cumulative_returns")
plt.plot(cumulativeret_xlk['benchmark'],label="benchmark returns")
plt.plot(cumulativeret_xlk['benchmark2'],label="benchmark returns2")
plt.plot(etf_xlk2['cum_ret'],label='ETF returns')
plt.legend(loc='upper left',fontsize=15)
plt.title('Technology: XLK',fontsize=20)

#%%



























