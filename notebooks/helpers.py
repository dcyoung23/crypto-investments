import pandas as pd
import talib as ta

# Forward looking window decision metrics
def window_metrics(df, window_size, min_periods, threshold):
    tp = str(window_size)+'d'
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
    # Close price at end of next day
    df['close_end_1d'] = df['close'].shift(-1)
    # Return for 1 day
    df['close_percent_change_1d'] = (df['close_end_1d'] - df['close']) / df['close']
    # Close price at end of window
    df['close_end_'+tp] = df['close'].shift(-window_size)
    # Lowest price during window
    df['close_min_'+tp] = df['low'].rolling(window=indexer, min_periods=min_periods).min()
    # Highest price during window
    df['close_max_'+tp] = df['high'].rolling(window=indexer, min_periods=min_periods).max()
    # Worst loss during window
    df['close_loss_'+tp] = (df['close_min_'+tp] - df['close']) / df['close']
    # Best gain during window
    df['close_gain_'+tp] = (df['close_max_'+tp] - df['close']) / df['close']
    # Close price change from begin to end of window
    df['close_change_'+tp] = df['close_end_'+tp] - df['close']
    # Return from start to end of window
    df['close_percent_change_'+tp] = df['close_change_'+tp] / df['close']
    # Buy logic
    df.loc[(df['close_change_'+tp] > 0) 
           #& (df['close_min_'+tp] >= df['close'])
           & (df['close_end_1d'] > df['close'])
           & (df['close_gain_'+tp] > threshold), 'decision'] = 'Buy'
    # Sell logic
    df.loc[(df['close_change_'+tp] <= 0) 
           #& (df['close_max_'+tp] <= df['close'])
           & (df['close_end_1d'] < df['close'])
           & (df['close_loss_'+tp] < -threshold), 'decision'] = 'Sell'
    df['decision'] = df['decision'].fillna('Hold')
    # Create dictionary for decision colors
    color_map = {'Buy': 'Green', 'Hold': 'Grey', 'Sell': 'Red'}
    # Create color column
    df['color'] = df['decision'].map(color_map)
    # Create binary columns to be used for correlation analysis
    df['buy'] = df['decision'].apply(lambda x: 1 if x == 'Buy' else 0)
    df['sell'] = df['decision'].apply(lambda x: 1 if x == 'Sell' else 0)
    df['hold'] = df['decision'].apply(lambda x: 1 if x == 'Hold' else 0)
    return df


def set_x_domain(df):
    return (df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d"))


def set_x_filter(df, start):
    return [start, df.index.max().strftime("%Y-%m-%d")]


def set_coin_name(df):
    return str(df['name'].unique()[0])


def set_transformation_cols(cols, transformation):
    return [col+'_'+transformation for col in cols]


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
    return df_out


# Momentum multiple slow and fast period metrics
def create_multi_tp_metrics(df, fastperiod, slowperiod, signalperiod):
    df_out = df.copy()
    df_out['apo'] = ta.APO(df_out['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=0)
    df_out['macd'], df_out['macdsignal'], df_out['macdhist'] = ta.MACD(df_out['close'], 
                                                                       fastperiod=fastperiod, 
                                                                       slowperiod=slowperiod, 
                                                                       signalperiod=signalperiod)
    df_out['ppo'] = ta.PPO(df_out['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=0)
    return df_out


def apply_transformation(df, cols, transformation, periods):
    df_out = df.copy()
    new_cols = set_transformation_cols(cols, transformation)
    if transformation == 'pct_change':
        df_out[new_cols] = df_out[cols].pct_change(periods)
    elif transformation == 'mean':
        df_out[new_cols] = df_out[cols].rolling(periods).mean()
    elif transformation == 'stdev':
        df_out[new_cols] = df_out[cols].rolling(periods).std()
    return df_out


def set_transformation_cols(cols, transformation):
    return [col+'_'+transformation for col in cols]


def create_rolling_r(df, target, cols, transformation, window, periods):
    df_out = apply_transformation(df, cols, transformation, periods)
    new_cols = set_transformation_cols(cols, transformation)
    return df_out[target].rolling(window).corr(df_out[new_cols])


