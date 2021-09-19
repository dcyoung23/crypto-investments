import numpy as np
import pandas as pd
import talib as ta
from ..utils.helpers import *

# Forward looking window decision metrics
def window_metrics(df, window_size, min_periods, threshold):
    df_out = df.copy()
    tp = str(window_size)+'d'
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
    # Close price at end of next day
    df_out['close_end_1d'] = df_out['close'].shift(-1)
    # Return for 1 day
    df_out['close_percent_change_1d'] = (df_out['close_end_1d'] - df_out['close']) / df_out['close']
    # Close price at end of window
    df_out['close_end_'+tp] = df_out['close'].shift(-window_size)
    # Lowest price during window
    df_out['close_min_'+tp] = df_out['low'].rolling(window=indexer, min_periods=min_periods).min()
    # Highest price during window
    df_out['close_max_'+tp] = df_out['high'].rolling(window=indexer, min_periods=min_periods).max()
    # Worst loss during window
    df_out['close_loss_'+tp] = (df_out['close_min_'+tp] - df_out['close']) / df_out['close']
    # Best gain during window
    df_out['close_gain_'+tp] = (df_out['close_max_'+tp] - df_out['close']) / df_out['close']
    # Close price change from begin to end of window
    df_out['close_change_'+tp] = df_out['close_end_'+tp] - df_out['close']
    # Return from start to end of window
    df_out['close_percent_change_'+tp] = df_out['close_change_'+tp] / df_out['close']
    # Buy logic
    df_out.loc[(df_out['close_change_'+tp] > 0) 
           & (df_out['close_end_1d'] > df_out['close'])
           & (df_out['close_gain_'+tp] > threshold), 'decision'] = 'Buy'
    # Sell logic
    df_out.loc[(df_out['close_change_'+tp] < 0) 
           & (df_out['close_end_1d'] < df_out['close'])
           & (df_out['close_loss_'+tp] < -threshold), 'decision'] = 'Sell'
    # All else hold
    #df_out['decision'] = df_out['decision'].fillna('Hold')
    df_out.loc[(pd.notnull(df_out['close_change_'+tp])) 
           & (pd.isnull(df_out['decision'])), 'decision'] = 'Hold'
    # Create dictionary for decision colors
    color_map = {'Buy': 'Green', 'Hold': 'Grey', 'Sell': 'Red'}
    # Create color column
    df_out['color'] = df_out['decision'].map(color_map)
    return df_out


# Single time period metrics
def create_single_tp_metrics(df, timeperiod):
    df_out = df.copy()
    df_out['adx'] = ta.ADX(df_out['high'], df_out['low'], df_out['close'], timeperiod=timeperiod)
    df_out['cci'] = ta.CCI(df_out['high'], df_out['low'], df_out['close'], timeperiod=timeperiod)
    df_out['cmo'] = ta.CMO(df_out['close'],  timeperiod=timeperiod)
    df_out['dx'] = ta.DX(df_out['high'], df_out['low'], df_out['close'], timeperiod=timeperiod)
    df_out['roc'] = ta.ROC(df_out['close'],  timeperiod=timeperiod)
    df_out['rsi'] = ta.RSI(df_out['close'],  timeperiod=timeperiod)
    df_out['willr'] = ta.WILLR(df_out['high'], df_out['low'], df_out['close'], timeperiod=timeperiod)
    df_out['atr'] = ta.ATR(df_out['high'], df_out['low'], df_out['close'], timeperiod=timeperiod)
    df_out['corr'] = ta.CORREL(df_out['high'], df_out['low'], timeperiod=timeperiod)
    df_out['linreg'] = ta.LINEARREG(df_out['close'], timeperiod=timeperiod)
    df_out['angle'] = ta.LINEARREG_ANGLE(df_out['close'], timeperiod=timeperiod)
    df_out['intercept'] = ta.LINEARREG_INTERCEPT(df_out['close'], timeperiod=timeperiod)
    df_out['slope'] = ta.LINEARREG_SLOPE(df_out['close'], timeperiod=timeperiod)
    df_out['stdev'] = ta.STDDEV(df_out['close'], timeperiod=timeperiod, nbdev=1)
    df_out['var'] = ta.VAR(df_out['close'], timeperiod=timeperiod, nbdev=1)
    df_out['tsf'] = ta.TSF(df_out['close'], timeperiod=timeperiod)
    df_out['bband_upper'], df_out['bband_middle'], df_out['bband_lower'] = ta.BBANDS(df_out['close'], timeperiod=timeperiod, matype=0)
    df_out['bband_relative'] = (df_out['close'] - df_out['bband_lower']) / (df_out['bband_upper'] - df_out['bband_lower'])
    return df_out


# Momentum multiple slow and fast period metrics
def create_multi_tp_metrics(df, fastperiod, slowperiod, signalperiod):
    df_out = df.copy()
    df_out['apo'] = ta.APO(df_out['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=0)
    df_out['macd'], df_out['macdsignal'], df_out['macdhist'] = ta.MACD(df_out['close'], fastperiod=fastperiod,
                                                                       slowperiod=slowperiod, signalperiod=signalperiod)
    df_out['ppo'] = ta.PPO(df_out['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=0)
    return df_out


# Social derived metrics
def create_social_metrics(df, custom_transform):
    df_out = df.copy()
    df_out['tweet_sentiment_bullish'] = (df_out['tweet_sentiment4'] + df_out['tweet_sentiment5'])
    df_out['tweet_sentiment_bearish'] = (df_out['tweet_sentiment1'] + df_out['tweet_sentiment2'])
    df_out['tweet_sentiment_net'] = df_out['tweet_sentiment_bullish'] - df_out['tweet_sentiment_bearish']
    df_out['tweet_sentiment_impact_bullish'] = (df_out['tweet_sentiment_impact4'] + df_out['tweet_sentiment_impact5'])
    df_out['tweet_sentiment_impact_bearish'] = (df_out['tweet_sentiment_impact1'] + df_out['tweet_sentiment_impact2'])   
    df_out['tweet_sentiment_impact_net'] = df_out['tweet_sentiment_impact_bullish'] - df_out['tweet_sentiment_impact_bearish']
    if custom_transform:
        for i in custom_transform:
            df_out = apply_transformation(df_out, i[0], i[1], i[2])
    return df_out