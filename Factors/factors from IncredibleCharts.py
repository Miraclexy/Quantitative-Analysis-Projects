# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:29:53 2018

@author: x.yi
"""

#==================================
# | factors from IncredibleCharts |
#=================================

import pandas as pd
import numpy as np
from tech.lib.signal.tech.volatility import atr
from tech.lib.functions.common import ema
from tech.lib.signal.tech.overlap import wma,sma
from tech.lib.signal.tech.moment import roc
import math
'''There we assume that we already have the function to compute ATR, EMA, WMA,
   SMA, ROC
'''

def atr_trailing_stop_signals(df,n=3):
    '''ATR Trailing Stop Signals
    Usage:
        Signals are used for exits:
        Exit your long position (sell) when price crosses below the ATR trailing stop line.
        Exit your short position (buy) when price crosses above the ATR trailing stop line.
        they can also be used to signal entries — in conjunction with a trend filter.
    steps:
        Calculate Average True Range ("ATR")
        Multiply ATR by your selected multiple — in our case 3 x ATR
        In an up-trend, subtract 3 x ATR from Closing Price and plot the 
        result as the stop for the following day.
        If price closes below the ATR stop, add 3 x ATR to Closing Price — to track a Short trade
        Otherwise, continue subtracting 3 x ATR for each subsequent day until 
        price reverses below the ATR stop.
        We have also built in a ratchet mechanism so that ATR stops cannot move 
        lower during a Long trade nor rise during a Short trade.
        The HighLow option is a little different: 3xATR is subtracted from the 
        daily High during an up-trend and added to the daily Low during a down-trend.
    '''
    res = df[['close','high','low']].copy()
    res['ATR'] = atr(res).ATR
    res['stop'] = res.close-n*res.ATR
    res['trade'] = np.nan
    res['highoption'] = np.nan
    res['lowoption'] = np.nan
    for i in range(len(res.close)):
        while res.close.iloc[i]>=res.stop.iloc[i]:
            res.close.iloc[i] = res.close.iloc[i]-n*res.ATR.iloc[i]
        res.trade.iloc[i] = res.close.iloc[i]+n*res.ATR.iloc[i]
        res.highoption.iloc[i] = res.high.iloc[i]-n*res.ATR.iloc[i]
        res.lowoption.iloc[i] = res.low.iloc[i]+n*res.ATR.iloc[i]
    return res
        
def chaikin_volitility(df,period=10):
    '''Chaikin Volitility
    Usage:
        Look for sharp increases in volatility prior to market tops and bottoms, 
        followed by low volatility as the market loses interest.
        A Chaikin Volatility peak occurs as the market retreats from a new high 
        and enters a trading range.
        The market ranges in a narrow band - note the low volatility.
        The breakout from the range is not accompanied by a significant rise in volatility.
        Volatility starts to rise as price rises above the recent high.
        A sharp rise in volatility occurs prior to a new market peak.
        The sharp decline in volatility signals that the market has lost impetus 
        and a reversal is likely.
    First, calculate an exponential moving average (normally 10 days) of the 
    difference between High and Low for each period: EMA [H-L]
    Next, calculate the percentage change in the moving average over a further 
    period (normally 10 days):
     ( EMA [H-L] - EMA [H-L 10 days ago] ) / EMA [ H-L 10 days ago] * 100
     '''
    res = df[['high','low']].copy()
    res['diff'] = res.high-res.low
    res['EMA'] = ema(res['diff'],period)
    res['EMA_10'] = ema(res['diff'].shift(10),period)
    res['Chaikin_Volitility'] = (res.EMA - res.EMA_10)/res.EMA_10*100
    return res
     
def coppock(df):
    '''Coppock Indicator
    Usage:
        to identify the commencement of bull markets.
        A bull market is signaled when the Coppock Indicator turns up from below zero.
    To calculate the Coppock Indicator:
        Calculate 14 month Rate of Change (Price) for the index. Use monthly closing price.
        Calculate 11 month Rate of Change (Price) for the index. Use monthly closing price.
        Add the results of 1 and 2.
        Calculate a 10 month weighted moving average of the result.
    '''
    res = df[['prices']].copy()
    res['month_rate_of_change_14'] = res['prices'].pct_change(14).shift(-14)
    res['month_rate_of_change_11'] = res['prices'].pct_change(11).shift(-11)
    res['temp'] = res['month_rate_of_change_14']+res['month_rate_of_change_11']
    res['coppock'] = wma(res['temp'],10)
    return res

def choppiness(df,n):
    '''Choppiness Index
    Choppiness Index = 100 * Log10{Sum(TrueRange,n) / [Maximum(TrueHigh,n) - 
    Minimum(TrueLow,n)]} / Log10(n)
    '''
    res = df[['high','low','close']].copy()
    res['sumTR'] = atr(res)['TR'].rolling(n).sum()
    res['maxtruehigh'] = res['high'].rolling(n).max()
    res['mintruelow'] = res['low'].rolling(n).min()
    res['choppiness'] = 100* (res.sumTR/(res.maxtruehigh-res.mintruelow)).apply(math.log,10) \
                /math.log(n,10)
    return res

    
def ease_of_movement(df):
    '''Ease of Movement
    Usage:
        Go long when Ease of Movement crosses to above zero (from below).
        Go short when Ease of Movement crosses to below zero (from above).
    The steps in calculating Ease of Movement are:
        Calculate the Mid-point for each day: (High + Low) / 2
        Calculate the Mid-point Move for each day: 
           Mid-point [today] - Mid-point [yesterday]
        The Box Ratio determines the ratio between height and width of the Equivolume box: 
           Volume [in millions] / (High - Low)
        Ease of Movement is then calculated as: Mid-point Move / Box Ratio
        Ease of Movement is normally smoothed with a 14 day exponential moving average.
     '''
    res = df[['high','low','volume']].copy()
    res['midpoint'] = (res.high+res.low)/2
    res['midmove'] = res.midpoint.diff()
    res['boxratio'] = res.volume/(res.high-res.low)
    res['ease_of_movement'] = res.midmove/res.boxratio
    res['ease_of_movement'] = ema(res['ease_of_movement'],14)
    return res
     
     
def heikin_ashi_candlesticks(df):
    '''Heikin-Ashi Candlesticks
    Signals should be interpreted in the same way as on traditional candlestick 
    charts. Long candles indicate a strong trend, while doji candles 
    (and spinning tops) indicate consolidation that may warn of a reversal. 
    
    Heikin-Ashi Candlesticks are calculated using smoothed values for Open, High, Low and Close:
        Heikin-Ashi Close is the average of Open, High, Low and Closing Price for the period.
        Heikin-Ashi Open is the average of the Heikin Ashi Open and Close for the previous candle.
        Heikin-Ashi High is the highest of three points for the current period: 
            The High
            Heikin-Ashi Open
            Heikin-Ashi Close
        Heikin-Ashi Low is the lowest of three points for the current period: 
            The Low
            Heikin-Ashi Open
            Heikin-Ashi Close
    '''
    return 1

def keltner_channels(df,n,halflife=14):
    '''Kelter Channels(Keltner Bands)
    The theory is that a movement that starts at one price band is likely to carry to the other.
    Go long when prices turn up at or below the lower band. Close your position 
    if price turns down near the upper band or crosses to below the moving average.
    Go short when price turns down at or above the upper band. Close your position 
    if price turns up near the lower band or crosses to above the moving average.
    
    Upper Band = Exponential MA of Closing Price + multiple of Average True Range
    Lower Band = Exponential MA of Closing Price - multiple of Average True Range
    '''
    res = df[['close','high','low']].copy()
    res['upper_band'] = ema(res['close'],halflife)+atr(res).ATR*n
    res['lower_band'] = ema(res['close'],halflife)-atr(res).ATR*n
    return res
     
def kst(df):
    '''KST Indicator
    First check whether price is trending. If the KST indicator is flat or stays 
    close to the zero line, the market is ranging and signals are unreliable.
        Go long when KST crosses above its signal line from below.
        Go short when KST crosses below the signal line from above.  
    formula:
        Calculate the Rate Of Change indicator for 9, 12, 18, and 24 months.
        Smooth the first three ROCs with a 6-Month Simple Moving Average.
        Smooth the 24-Month ROC with a 9-Month Simple Moving Average.
        Weight the four Smoothed ROCs according to their period.
        The sum of the four periods is 63 months, so values for the 9-Month 
            Smoothed ROC will be multiplied by 9/63, the 12-Month ROC by 12/63, and so on...
        Sum the weighted values to calculate the KST indicator.
        Calculate the signal line using a 9-Month Simple Moving Average of KST.
    '''
    res = df[['prices']].copy()
    res['ROC9'] = roc(res['prices'],9).ROC
    res['ROC12'] = roc(res['prices'],12).ROC
    res['ROC18'] = roc(res['prices'],18).ROC
    res['ROC24'] = roc(res['prices'],24).ROC
    res['smoothROC9'] = sma(res['ROC9'],6)
    res['smoothROC12'] = sma(res['ROC12'],6)
    res['smoothROC18'] = sma(res['ROC18'],6)
    res['smoothROC24'] = sma(res['ROC24'],9)
    res['WeightedROC9'] = wma(res['smoothROC9'],9)
    res['WeightedROC12'] = wma(res['smoothROC12'],12)
    res['WeightedROC18'] = wma(res['smoothROC18'],18)
    res['WeightedROC24'] = wma(res['smoothROC24'],24)    
    res['adjustsmoothROC9'] = res['smoothROC9']*(9/63)
    res['adjustsmoothROC12'] = res['smoothROC12']*(12/63) 
    res['adjustsmoothROC18'] = res['smoothROC18']*(18/63)
    res['adjustsmoothROC24'] = res['smoothROC24']*(24/63)
    res['KST'] = res['WeightedROC9']+ res['WeightedROC12'] + \
            res['WeightedROC18'] + res['WeightedROC24']
    res['signal_line'] = sma(res['KST'],9)
    return res
     
     
def hmv(df,period):
    '''Hull Moving Average
    Usage:
        Alan Hull recommends using his moving average for directional signals 
        and not for crossovers which could be distorted by the lag.
        Go long when Hull Moving Average turns up; 
        Go short when Hull Moving Average turns down.
    Alan Hull uses three Weighted Moving Averages (WMA) in his formula:
        Calculate the WMA for the Period (e.g. 13 Weeks).
        Divide the Period by 2 and use the Integer value to calculate a second WMA.
        Multiply the second WMA by 2 then subtract the first WMA.
        Calculate the Square Root of the Period and take the Integer value.
        Use the resulting Integer value to calculate a third WMA of the result from the first two WMAs.
    Here is more mathematical notation for n periods:
        WMA(Integer(SQRT(n)),WMA(2*Integer(n/2),data) - WMA(n,data))
    '''
    res = df[['prices']].copy()
    res['WMA1'] = wma(res['prices'],period)
    res['WMA2'] = wma(res['prices'],int(period/2))
    res['temp'] = 2*res.WMA2-res.WMA1
    res['HMV'] = wma(res['temp'],int(np.sqrt(period)))
    return res
    
def mao(df,period):
    '''Moving Average Osillator
    Usage:
        Go long on a bullish divergence where the second dip does not cross below -50%.
        Go long when a downward trendline on the Moving Average Oscillator is broken 
        and the Oscillator crosses to above zero.
        Exit long positions if the Moving Average Oscillator turns down while above 50%.
        Go short on a bearish divergence where the second peak does not cross above 50%.
        Go short when a downward trendline on the Moving Average Oscillator is 
        broken and the Oscillator crosses to below zero.
        Exit short positions if the Moving Average Oscillator turns up while below -50%.
    The formula is simply :
        (Close - Exponential MA) / Exponential MA expressed as a percentage.
    '''
    res = df[['close']].copy()
    res['MAO'] = (res[['close']]-ema(res[['close']],period))/ema(res[['close']],period)
    def to_percentage(x):
        return str(round(x*100,4)) + '%'
    res['MAO'] = res.MAO.apply(to_percentage)
    return res

def macd_indicator(df):
    '''MACD Indicator
    The MACD indicator is calculated as the difference between the fast and slow moving averages:
        MACD = 12-Day exponential moving average minus 26-Day exponential moving average
        The signal line is calculated as a 9-day exponential moving average of MACD.
    '''
    res = df[['close']].copy()
    res['MACD'] = ema(res[['close']],12)-ema(res[['close']],26)
    res['signal_line'] = ema(res[['MACD']],9)
    return res
     
def mass_index(df):
    '''Mass Index
    Usage:
        Go long if there is a reversal bulge and EMA points downward.
        Go short if there is a reversal bulge and EMA points upward.
    The default settings are:
        indicator window - 25 days
        upper level - 27.0
        lower level - 26.5
    To calculate the Mass Index:
        1.Calculate the range for each period: High - Low
        2.Calculate a 9 day exponential moving average of the range: EMA [H - L]
        3.Calculate a 9 day exponential moving average of the above: 
           EMA ( EMA [H - L] )
        4.Divide the first: EMA by the second: EMA [H - L] / EMA ( EMA [H - L] )
        5.Add the values for the selected number of periods (normally 25).
    '''
    res = df[['high','low']].copy()
    res['range'] = res.high-res.low
    res['EMA'] = ema(res['range'],9)
    res['EMAEMA'] = ema(res['EMA'],9) 
    res['mass'] = res.EMA-res.EMAEMA
    res['mass'] = res['mass']+25
    res['upperlevel'] = 27.0
    res['lowerlevel'] = 26.5
    return res

def pts(df):
    '''Percentage Trailing Stops
    Usage:
        The signals are used for exits.
        Exit your long position when price crosses below the Percentage Trailing Stop line.
        Exit your short position when price crosses above the Percentage Trailing Stop line.
    Trailing stops are normally calculated using closing prices:
        1.In an up-trend, subtract 10 percent from the Closing Price and plot the 
            result as the stop for the following day
        2.If price closes below trailing stop, add 10 percent to the Closing 
            Price — to track a Short trade
        3.Otherwise, continue subtracting 10 percent from Closing Price each day 
            until price reverses below the trailing stop
        4.There is also built in a ratchet mechanism so that ATR stops do not 
            move lower during a Long trade nor rise during a Short trade.
        5.If you select the High/Low option, the trailing stop is calculated 
            using the daily High in an up-trend — and the daily Low in a down-trend.
    '''
    res = df[['close']].copy()
    res['trailingstop'] = 0.9*res['close'].shift(1)
    res['trade'] = np.nan
    for i in range(len(res)):
        while res.close.iloc[i]>=res.trailingstop.iloc[i]:
            res.close.iloc[i] = 0.9*res.close.iloc[i]    
        res.trade.iloc[i] = res.close.iloc[i]*1.1
    return res

def pp(df):
    '''Pivot Points
    Several methods of calculating Pivot Points have evolved over the years, 
    but we will stick to the Standard Method:
        1.The Pivot Point is the average of the previous High, Low and Closing Price.
        2.Resistance level 2 (R2) = Pivot Point + High - Low
        3.Resistance level 1 (R1) = Pivot Point x 2 - Low
        4.Support level 1 (S1) = Pivot Point x 2 - High
        5.Support level 2 (S2) = Pivot Point - High + Low
    '''
    res = df[['high','low','close']].copy()
    res['pivotpoint'] = ((res.high+res.low+res.close)/3).shift(1)
    res['R2'] = res.pivotpoint+res.high-res.low
    res['R1'] = 2*res.pivotpoint-res.low
    res['S1'] = res.pivotpoint*2-res.high
    res['S2'] = res.pivotpoint-res.high+res.low
    return res     
    
def pvt(df):
    '''Price and Volume Trend(PVT)
    Usage:
    During a ranging market watch for a rising or falling Price and Volume Trend.
        Rising PVT signals an upward breakout.
        Falling PVT signals a downward breakout.
    Trending Market
        A rising Price and Volume Trend confirms an up-trend and a falling PVT confirms a down-trend.
        Bullish divergence between PVT and price warns of market bottoms.
        Bearish divergence between PVT and price warns of market tops.
    The steps in the Price and Volume Trend calculation are:
        1.Calculate the Percentage Change in closing price: 
           ( Closing Price [today] - Closing Price [yesterday] ) / Closing Price [yesterday]
        2.Multiply the Percentage Change by Volume: 
           Percentage Change * Volume [today]
        3.Add to yesterday's cumulative total: 
           Percentage Change * Volume [today] + PVT [yesterday]
    '''
    res = df[['close','volume']].copy()
    res['pct'] = res['close'].pct_change(1)
    res['pvt'] = res.close*res.volume
    res['pvt'] = res.pvt.shift(1)+res.pct*res.volume
    return res
    
def pb(df):
    '''Percentage Bands
    Usage:
    The signals may be used for entries and exits. In an up-trend:
        Go Long when price closes above the upper band.
        Exit the long position when price closes below the lower band.
    In a down-trend:
        Go Short when price closes below the lower band.
        Exit the short position when price closes above the upper band.
    Percentage Bands are normally calculated using closing prices:
        Add and subtract the selected percent (normally 10%) from the Closing 
            Price and plot the result as the band for the following day
        There is also built in a ratchet mechanism so that the lower band does 
            not move lower during a Long trade nor rise during a Short trade.
    If you select the High/Low option, the bands are calculated from the 
    High and Low prices rather than the Close:
        The lower band is calculated by subtracting the selected percent (normally 10%) from the High.
        The upper band is calculated by adding the selected percent (normally 10%) to the Low.
    '''
    res = df[['close']].copy()
    res['upband'] = (res.close*1.1).shift(1)
    res['lowband'] = (res.close*0.9).shift(1)
    return res

def safezone(df,period):
    '''Safezone Formula
    Define the Trend
        First compare Closing Price to an exponential moving average to define the trend.
        If Closing Price is above the moving average for the selected period, 
        that means that the trend (and the MA slope) is upward.
        If Closing Price is below the moving average, the trend is downward. 
    Directional Movement
        The second element is Directional Movement. This is calculated in a 
        similar fashion to DI+ and DI- in the Directional Movement System:
        +DM = Today's High - Yesterday's High (when price moves upward)
        -DM  = Yesterday's Low - Today's Low (when price moves downward)
        The difference is that you can have both +DM and -DM on the same day. 
        If there is an outside day then both calculations will be positive. 
        For an inside day both calculations are zero.
   Directional Movement Days
       Calculate the number of days with +DM in the selected period; and the 
       number of -DM days. Elder uses the same selected period for Directional 
       Movement as he does for the moving average, but there appears to be no 
       reason why this could not be varied. 
   When the Trend is UP
       Calculate -DM Average:
           Sum of -DM for the period / Number of -DM days
           Then calculate the Stop Level for today:
           Today's Stop = Yesterday's Low - 2.5 * -DM Average
           To delay/prevent the stop from being lowered, use the maximum of the last 3 days' stops. 
   When the Trend is DOWN
       Calculate +DM Average:
           Sum of +DM for the period / Number of +DM days
           Then calculate the Stop Level for today:
           Today's Stop = Yesterday's High + 2.5 * +DM Average
           To delay/prevent the stop from being raised, take the minimum of the last 3 days' stops.
           Note: We use a multiple of 2.5 in the above example, but any multiple between 2 and 4 is acceptable. 
    '''
    res = df[['close','high','low']].copy()
    res['ema'] = ema(res['close'],period)
    res['+DM'] = res.high.diff()
    res['-DM'] = -1*res.low.diff()
    negative_DM_avg = res['-DM'].mean()
    positive_DM_avg = res['+DM'].mean()   
    for i in range(len(res)):
        if res.close.iloc[i]>res.ema.iloc[i]:
            res['todaystop'] = res.high.shift(1)+2.5*positive_DM_avg
        else:
            res['todaystop'] = res.low.shift(1)-2.5*negative_DM_avg
    return res
    
  
    
    
    
    
    
    
    
    
    