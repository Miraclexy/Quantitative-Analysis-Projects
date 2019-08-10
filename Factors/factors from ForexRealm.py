# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:16:21 2018

@author: x.yi
"""

#===========================
#| Factors from ForexRealm |
#==========================

import pandas as pd
import numpy as np
from tech.lib.signal.tech.volatility import atr 
from tech.lib.signal.tech.overlap import smma,sma
'''There we assume that we already have the function to compute ATR, SMMA, SMA'''

def abi(df,yesterdaydate,todaydate):
    '''Absolute Breadth Index(ABI or ABX)
    u: the number of companies whose stock prices go up
    d: the number of companies whose stock prices go dowm
    (ABI)= 100×∣u-d∣/∣u+d∣
    The higher the ABI factor, the more the difference between the number of 
    companies whose stock prices go up and those whose stock prices go down.

    parameters:
        df includes date，asset，dataframe of stock prices
        yesterdaydate: the day before we want to compute ABI on
        todaydate: the day we want to compute ABI on 
    '''
    res = pd.DataFrame(index=df.asset[df.date==yesterdaydate])
    res['yesterdayprices'] = df[['asset','prices']][df.date==yesterdaydate] \
            .set_index(df.asset[df.date==yesterdaydate]).prices
    res['todayprices'] = df[['asset','prices']][df.date==todaydate] \
            .set_index(df.asset[df.date==todaydate]).prices
    res['status'] = res.todayprices-res.yesterdayprices
    res.status = res.status.apply(lambda x : 1 if x>=0 else -1)
    advance = res.status[res.status==1].sum()
    decline = abs(res.status[res.status==-1].sum())
    abi = 100*abs((advance-decline))/abs((advance+decline))
    return pd.DataFrame(np.array([abi,advance,decline]).reshape(1,3),index=[todaydate], \
                        columns=['ABI','advance','decline'])

def asi(df,T=300):
    '''Accumulative Swing Indicator(ASI)
    ASI = 50*(Cy-C+0.5*(Cy-Oy)+0.25*(C-O))/R*(K/T)
    parameters:
        C = Today's closing price
        Cy = Yesterday's closing price
        Hy = Yesterday's highest price
        K = The greatest of: Hy - C and Ly - C
        L = Today's lowest price
        Ly = Yesterday's lowest price
        O = Today's opening price
        Oy = Yesterday's opening price
        R = This varies based on relationship between today's closing price 
            and yesterday's high and low prices
        T = the maximum price changing during trade session
    Usage:
        ASI has positive value — uptrend.
        ASI has negative value — downtrend.
        ASI trend line breakout — validates a breakout on the price chart
    '''
    #SI(i) = (50*(close[i-1]-close[i]+0.5*(close[i-1]-open[i-1]) 
    #       +0.25*(close[i]-open[i]))/R)*(K/T)
    #ASI(i) = SI(i-1)+SI(i)
    res = df[['close','open','high','low']].copy()
    res['temp1'] = -1*res.close.diff()
    res['temp2'] = (res.close-res.open).shift()
    res['temp3'] = res.close-res.open
    #K=max((close[i-1]-close[i]),(low[i-1]-close[i]))
    res['K'] = np.nan
    for i in range(len(res)-1):
        if res.close.iloc[i]>=res.low[i]:
            res.K.iloc[i+1] = res.close.iloc[i]-res.close.iloc[i+1]
        else:
            res.K.iloc[i+1] = res.low.iloc[i]-res.close.iloc[i+1]
    '''
     R = TR - 0.5*ER + 0.25*SH
     TR = atr(df,period=14)
     if(Close[i+1] >= Low[i] && Close[i+1] <= High[i]) 
           ER = 0; 
       else 
         {
           if(Close[i+1] > High[i]) 
               ER = MathAbs(High[i] - Close[i+1]);
           if(Close[i+1] < Low[i]) 
               ER = MathAbs(Low[i] - Close[i+1]);
         }

      SH = MathAbs(Close[i+1] - Open[i+1]);
      
     '''
     
    res['TR'] = atr(df)[['ATR']]
    res['ER'] = np.nan
    for i in range(len(res)-1):
        if res.close.iloc[i+1]>=res.low.iloc[i] and \
            res.close.iloc[i+1]<=res.high.iloc[i]:
                res.ER.iloc[i]=0
        else:
            if res.close.iloc[i+1]>res.high.iloc[i]:
                res.ER.iloc[i]=abs(res.high.iloc[i]-res.close.iloc[i+1])
            elif res.close.iloc[i+1]<res.low.iloc[i]:
                res.ER.iloc[i]=abs(res.low.iloc[i]-res.close.iloc[i+1])
    res['SH'] = abs(res.close.shift(-1)-res.open.shift(-1))
    res['R'] = res.TR-0.5*res.ER+0.25*res.SH
    res['SI'] = (50*(res.temp1+0.5*res.temp2+0.25*res.temp3)/res.R)*(res.K/T)
    res['ASI'] = np.nan
    for i in range(len(res)-1):
        res.ASI.iloc[i+1]=res.SI.iloc[i+1]+res.SI.iloc[i]  
        
    return res.ASI
        
def alligator(df):
    '''Alligator Indicator
    Alligator indicator consists of 3 Moving averages:
    Alligator’s jaws:13-period Simple Moving Average built from (High+Low)/2, 
                     moved into the future by 8 bars;
    Alligator’s teeth:8-period Simple Moving Average built from (High+Low)/2, 
                      moved by 5 bars into the future;
    Alligator’s lips:5-period Simple Moving Average built from (High+Low)/2, 
                     moved by 3 bars into the future.
    Usage:
        uptrend:jaws at the bottom, then teeth, then lips on top.
        downtrend:jaws on top, teeth, then lips below.
    '''
    res = df[['prices']].copy()
    res['jaws'] = smma(res.prices,period=13,shift=8)
    res['teeth'] = smma(res.prices,period=8,shift=5)
    res['lips'] = smma(res.prices,period=5,shift=3)
    return res
        
def chandlier_exit(df):
    '''Chandlier Exit
    Make profits in the direction of trend development, 
    calculate the stock price at the time of exit
    uptrend = Highest High(in the past 22 days)-3*ATR(for 22 days)
    downtrend = Lowest Low(in the past 22 days)+3*ATR(for 22 days)
    '''
    res = df[['high','low','close']].copy()
    res['uptrend'] = (res[['high']].rolling(22).max()).high-3*(atr(df,period=22).ATR)
    res['downtrend'] = (res[['low']].rolling(22).min()).low+3*(atr(df,period=22).ATR)
    return res
    
def acceleration_bands(df,Factor=0.001):
    '''Acceleration Bands Indicator(AB indicator)
    Usage:
        Breakout outside Acceleration bands suggest a beginning of a strong 
        rally or a sell-off.
        Closing inside the bands afterward signals about the end of a rally or 
        a sell-off.    
        Acceleration Bands principal use is in finding the acceleration in 
        currency pair price and benefit as long as this acceleration preserves.
        2 consecutive closes outside Acceleration Bands suggest an entry point 
        in the direction of the breakout. Then position is kept till the first 
        close back inside the Bands.
    '''
#Upperband = ( High * ( 1 + 2 * (((( High - Low )/(( High + Low ) / 2 )) * 1000 ) * Factor )));
#Lowerband = ( Low * ( 1 - 2 * (((( High - Low )/(( High + Low ) / 2 )) * 1000 ) * Factor )));
#Factor=0.001 
    res = df[['high','low']].copy()
    res['Upperband'] = (res.high*(1+2*((((res.high-res.low)/((res.high+res.low)  \
                       /2 ))*1000 )*Factor)))   
    res['Lowerband'] = (res.low*(1-2*((((res.high-res.low)/((res.high+res.low)  \
                       /2))*1000)*Factor)))
    return res
        
def chaikin_money_flow(df,period=21):
    '''Chaikin Money Flow(CMF)
    CMF = sum(((( C-L )-( H-C )) / ( H-L ))*V ) / sum(V) 
    Where:
        C- close
        L - low
        H - high
        V - volume (21 period)
    Usage:
        if CMF is positive, the market is strong;
        if CMF is negative, the market is weak.
    '''
    res = df[['close','low','high','volume']].copy()
    res['CMF'] = ((((res.close-res.low)-(res.high-res.close))/ \
                   (res.high-res.low))*res.volume).rolling(21).sum()/(res[['volume']].rolling(21) \
                       .sum()).volume
    return res

def csi(df,V,M,C):
    '''Commodity Selection Index(CSI)
    CSI suggest that the best commodities are:
        - high in directional movement (DMI indicator value)
        - high in volatility (Volatility Index value and ATR)
        - have reasonable margin requirements (relative to directional movement & volatility)
        - have reasonable commission rates 
    Usage:
        Trade with high CSI values
    CSI = ADXR*ATR(14)(V/sqrt(M)*1/(150+C))*100
    where:
        ADXR:Average Directional Movement Index Rating
        ATR(14):14-days Average True Range
        V:value of a 1 $ move(or the basic increment of ATR(14) in dollar)
        M:margin requirement in dollars
        C:commition in dollars
    Note: the result of the term 1/(150+C) must be carried to four decimal places
    ADXR = (ADT(today)+ADX(14 days ago))/2
    '''
    res = df[['high','low','close']].copy()
    res['ATR'] = atr(res,period=14)[['ATR']]
    res['ATR14'] = atr(res.shift(14),period=14)[['ATR']]
    res['ADXR'] = (res.ATR+res.ATR14)/2
    res['CSI'] = res.ADXR*res.ATR*(V/np.sqrt(M)*1/(150+C))*100
    return res[['CSI','ATR','ATR14','ADXR']]
    
def adr(df,yesterdaydate,todaydate):
    '''Advance Decline Ratio(ADR)
    Advance/Decline Ratio = Number of advancing moments / Number of declining moments
    Usage:
        ADR rising and so does the price — healthy trend.
        ADR falling and so does the price — healthy trend.
        ADR reading diverge from the price — trend may change.
        ADR crossing above 1.00 level — an uptrend has been established
        ADR crossing below 1.00 level — a downtrend has been established.
        The further ADR moves from 1.00 level the more mature current trend is.         
    '''
    res = df.copy()
    res = abi(res,yesterdaydate,todaydate)
    res['ADR'] = res.advance-res.decline
    return res

def dpo(df,n,period=20):
    '''Detrended Price Oscillator(DPO)
    Usage:
        Buy when DPO hits zero from above or dips below zero for a while 
        and then goes up above zero.
        Sell when DPO hits zero level from below or even crosses above zero 
        for a while and then turns back below zero. 
        Buy after DPO dips below an oversold zone and then exits from it 
        closing above the oversold zone.
        Sell after Detrended Price Oscillator enters an overbought zone and 
        then exits from it and closes below the overbought zone. 
    DPO=close-Moving Average((n/2)+1 days ago)
    '''
    res = df[['close']].copy()
    res['MA'] = (res.close.shift(int((n/2)+1))).rolling(period).mean()
    res['DPO'] = res.close-res.MA
    return res
    
def awesome_osillator(df):
    '''Awesome Osillator Indicator(AO)
    Awesome Oscillator shows the difference in between the 5 SMA and 34 SMA.
    If to be precise, 5 SMA of midpoints is subtracted from 34 SMA of midpoints 
    which allows to see the market momentum.
    Usage:
        When AO is above zero - only Buy orders should be taken.
        When AO is below zero - only Sell orders should be taken.
    Awesome Oscillator = SMA (MEDIAN PRICE, 5) - SMA (MEDIAN PRICE, 34)
    MEDIAN PRICE = (HIGH+LOW)/2
    Where: SMA - Simple Moving Average 
    '''
    res = df[['high','low']].copy()
    res['median'] = (res.high+res.low)/2
    res['AO'] = sma(res['median'],5)-sma(res['median'],34)
    return res


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        