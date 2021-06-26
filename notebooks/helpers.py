import pandas as pd

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