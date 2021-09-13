import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import talib as ta
from ..utils.helpers import *
from ..features.make_features import *

# Plot to show decision points on Close line chart - Matplotlib/Seaborn version with legend
def decision_plot(df, ax):
    ax.plot(df['close'])
    buy = df[df['decision'] == 'Buy']
    hold = df[df['decision'] == 'Hold']
    sell = df[df['decision'] == 'Sell']
    sns.scatterplot(x=buy.index, y=buy['close'], color='tab:green',  marker='o', s=50, label='Buy', ax=ax)
    sns.scatterplot(x=hold.index, y=hold['close'], color='tab:gray',  marker='o', s=50, label='Hold', ax=ax)
    sns.scatterplot(x=sell.index, y=sell['close'], color='tab:red',  marker='o', s=50, label='Sell', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend(loc='best')
    plt.suptitle(str(df['name'].unique()[0]))
    plt.tight_layout()

# Custom Plotly visualization single subplot with slider and selectors
def single_slider(traces, params):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=params['vertical_spacing'],
                        row_heights=params['row_heights'],
                        subplot_titles=params['subplot_titles']
                       )
    for trace in traces:
        if trace['mode'] == 'markers':
            fig.add_trace(go.Scatter(mode=trace['mode'],
                                     x=trace['y'].index, 
                                     y=trace['y'],
                                     name=trace['name'],
                                     marker=trace['marker'],
                                     marker_color=trace['marker_color'],
                                     showlegend=trace['showlegend']
                                    )
                         )
        else:
            fig.add_trace(go.Scatter(x=trace['y'].index,
                                     y=trace['y'],
                                     name=trace['name'],
                                     line=trace['line']
                                    )
                         )
    
    # Add range slider and selector
    fig.update_xaxes(matches='x',
                     range=params['x_domain'],
                     rangeslider= {'visible': True, 'range': params['x_domain']},
                     rangeselector=dict(buttons=list([
                         dict(count=20, label='20d', step='day', stepmode='backward'),
                         dict(count=50, label='50d', step='day', stepmode='backward'),
                         dict(count=100, label='100d', step='day', stepmode='backward'),
                         dict(step='all')])
                                       )
                    )

    fig.update_layout(showlegend=params['showlegend'],
                      title_text=params['title_text'],
                      width=params['fig_size'][0], height=params['fig_size'][1])
    xrange = params['xrange']
    if xrange:
        fig.update_layout(xaxis_range=xrange)
    fig.show(params['renderer'])


# Custom Plotly visualization stacked subplot with slider and selectors
def stacked_slider(traces, params):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=params['vertical_spacing'],
                        row_heights=params['row_heights'],
                        subplot_titles=params['subplot_titles']
                       )
    for trace in traces:
        if trace['mode'] == 'markers':
            fig.add_trace(go.Scatter(mode=trace['mode'],
                                     x=trace['y'].index, 
                                     y=trace['y'],
                                     name=trace['name'],
                                     marker=trace['marker'],
                                     marker_color=trace['marker_color'],
                                     showlegend=trace['showlegend']
                                    ),
                          row=trace['row'], col=1
                         )
        elif trace['mode'] == 'bar':
            fig.add_trace(go.Bar(x=trace['y'].index,
                                 y=trace['y'],
                                 name=trace['name'],
                                 marker_color=trace['marker_color'],
                                 showlegend=trace['showlegend']
                                 ),
                          row=trace['row'], col=1
                         )            
        else:
            fig.add_trace(go.Scatter(x=trace['y'].index,
                                     y=trace['y'],
                                     name=trace['name'],
                                     line=trace['line'],
                                     showlegend=trace['showlegend']
                                    ),
                          row=trace['row'], col=1
                         )
    # Add horizontal lines
    for hline in params['hlines']:
        fig.add_hline(y=hline['y'], 
                      line_width=hline['line_width'],
                      line_dash=hline['line_dash'], 
                      line_color=hline['line_color'],
                      row=hline['row'])
        
    # Add range slider and selector
    fig.update_xaxes(matches='x',
                     range=params['x_domain'],
                     rangeselector=dict(buttons=list([
                         dict(count=20, label='20d', step='day', stepmode='backward'),
                         dict(count=50, label='50d', step='day', stepmode='backward'),
                         dict(count=100, label='100d', step='day', stepmode='backward'),
                         dict(step='all')])
                                       )
                    )

    # Turn off rangeslider for top plot
    fig.update_xaxes(rangeslider= {'visible': False, 'range': params['x_domain']}, row=1, col=1)
    # Turn on rangeslider/off rangeselector for bottom plot
    fig.update_xaxes(rangeslider= {'visible': True, 'range': params['x_domain']},
                     rangeselector= {'visible': False}, row=2, col=1)
    
    fig.update_layout(showlegend=params['showlegend'],
                      title_text=params['title_text'],
                      width=params['fig_size'][0], height=params['fig_size'][1])
    xrange = params['xrange']
    if xrange:
        fig.update_layout(xaxis_range=xrange)
    fig.show(params['renderer'])


def bband_plot(df, timeperiod, matype, start, renderer):
    coin = set_coin_name(df)
    x_domain = set_x_domain(df)
    x_filter = set_x_filter(df, start)
    bband_upper, bband_middle, bband_lower = ta.BBANDS(df['close'], timeperiod=timeperiod, matype=matype)

    params = dict(vertical_spacing=0.1, row_heights=[1], subplot_titles=(None), 
                        x_domain=x_domain, showlegend=True,
                        title_text=coin+' Bollinger Bands with Buy-Hold-Sell Points',
                        fig_size=(800, 600), xrange=x_filter, renderer=renderer)

    traces = [dict(mode='lines', x=df.index, y=df['close'], name='Close', 
                   line={'color': 'CornFlowerBlue'}),
              dict(mode='lines', x=bband_upper.index, y=bband_upper, name='Upper Band',
                   line={'color': 'DarkGrey', 'dash': 'solid'}),
              dict(mode='lines', x=bband_middle.index, y=bband_middle, name='Middle Band',
                   line={'color': 'DarkGrey', 'dash': 'dash'}),
              dict(mode='lines', x=bband_lower.index, y=bband_lower, name='Lower Band',
                   line={'color': 'DarkGrey', 'dash': 'solid'}),
              dict(mode='markers', x=df.index, y=df['close'], name=None, 
                   marker={'size': 4}, marker_color=df['color'], showlegend=False)
             ]

    single_slider(traces, params)


def ma_plot(df, timeperiod, ma, exp, start, renderer):
    coin = set_coin_name(df)
    x_domain = set_x_domain(df)
    x_filter = set_x_filter(df, start)
    sma = ta.SMA(df['close'], timeperiod=timeperiod)
    wma = ta.WMA(df['close'], timeperiod=timeperiod)
    ema = ta.EMA(df['close'], timeperiod=timeperiod)
    dema = ta.DEMA(df['close'], timeperiod=timeperiod)
    tema = ta.TEMA(df['close'], timeperiod=timeperiod)

    params = dict(vertical_spacing=0.1, row_heights=[1], subplot_titles=(None), 
                        x_domain=x_domain, showlegend=True,
                        title_text=coin+' Moving Averages (MA) with Buy-Hold-Sell Points',
                        fig_size=(800, 600), xrange=x_filter, renderer=renderer)
    
    traces = [dict(mode='lines', x=df.index, y=df['close'], name='Close', 
                   line={'color': 'CornFlowerBlue'}),
              dict(mode='markers', x=df.index, y=df['close'], name=None, 
                   marker={'size': 4}, marker_color=df['color'], showlegend=False)
             ]
    
    ma_traces = [dict(mode='lines', x=sma.index, y=sma, name='Simple MA (SMA)',
                      line={'dash': 'solid'}),
                 dict(mode='lines', x=wma.index, y=wma, name='Weighted MA (WMA)',
                      line={'dash': 'solid'})]
    ema_traces = [dict(mode='lines', x=ema.index, y=ema, name='Exponential MA (EMA)',
                      line={'dash': 'solid'}),
                 dict(mode='lines', x=dema.index, y=dema, name='Double EMA (DEMA)',
                      line={'dash': 'solid'}),
                 dict(mode='lines', x=tema.index, y=tema, name='Triple EMA (TEMA)',
                      line={'dash': 'solid'})]

    if ma:
        traces.extend(ma_traces)
    if exp:
        traces.extend(ema_traces)

    single_slider(traces, params)


def macd_plot(df, fastperiod, slowperiod, signalperiod, start, renderer):
    coin = set_coin_name(df)
    x_domain = set_x_domain(df)
    x_filter = set_x_filter(df, start)
    macd, macdsignal, macdhist = ta.MACD(df['close'], 
                                         fastperiod=fastperiod, 
                                         slowperiod=slowperiod, 
                                         signalperiod=signalperiod)
    
    macdhist_color = pd.DataFrame(macdhist, columns=['macdhist'])
    macdhist_color['color'] = np.where(macdhist_color["macdhist"] < 0, 'red', 'green')

    params = dict(vertical_spacing=0.1, row_heights=[1, .75], subplot_titles=(None, 'MACD'), 
                  x_domain=x_domain, showlegend=True,
                  title_text=coin+' Buy-Hold-Sell Points with MACD',
                  hlines=[],
                  fig_size=(800, 600), xrange=x_filter, renderer=renderer)

    traces = [dict(mode='lines', x=df.index, y=df['close'], name='Close', 
                   line={'color': 'CornFlowerBlue'}, showlegend=True, row=1),
              dict(mode='markers', x=df.index, y=df['close'], name=None, 
                   marker={'size': 4}, marker_color=df['color'], showlegend=False, row=1),
              dict(mode='lines', x=macd.index, y=macd, name='MACD',
                   line={'dash': 'solid'}, showlegend=True, row=2),
              dict(mode='lines', x=macdsignal.index, y=macdsignal, name='MACD Signal',
                   line={'dash': 'solid'}, showlegend=True, row=2),
              dict(mode='bar', x=macdhist.index, y=macdhist, name='MACD Histogram',
                   marker_color=macdhist_color['color'], showlegend=True, row=2) 
             ]

    stacked_slider(traces, params)


# Generic stacked Buy-Hold-Sell (bhs) plot
def generic_bhs_plot(df, custom_params, start, renderer):
    df_out = df.copy()
    coin = set_coin_name(df)
    col = custom_params['col']
    label = col
    x_domain = set_x_domain(df)
    x_filter = set_x_filter(df, start)
    sp = custom_params['sp']
    if sp:
        df_out = create_single_tp_metrics(df_out, 
                                          sp['timeperiod'])
    mp = custom_params['mp']
    if mp:
        df_out = create_multi_tp_metrics(df_out, 
                                          mp['fastperiod'], 
                                          mp['slowperiod'], 
                                          mp['signalperiod'])
    
    params = dict(vertical_spacing=0.1, row_heights=[1, .5], subplot_titles=(None, label), 
                  x_domain=x_domain, showlegend=True,
                  title_text=coin+' Buy-Hold-Sell Points with '+label,
                  hlines=custom_params['hlines'], fig_size=(800, 600), xrange=x_filter, renderer=renderer)

    traces = [dict(mode='lines', x=df.index, y=df['close'], name='close', 
                   line={'color': 'CornFlowerBlue'}, showlegend=True, row=1),
              dict(mode='markers', x=df.index, y=df['close'], name='color', 
                   marker={'size': 4}, marker_color=df['color'], showlegend=False, row=1),
              dict(mode='lines', x=df_out.index, y=df_out[col], name=label,
                   line={'dash': 'solid'}, showlegend=True, row=2) 
             ]

    stacked_slider(traces, params)


def global_corr_heatmap(df, transform_cols, cols, method, ax):
    df_out = df[transform_cols].pct_change()
    if cols:
        # Add in the designated cols that are not being transformed
        df_out[cols] = df[cols]
    coin = set_coin_name(df)
    corr = df_out.corr(method=method)
    sns.heatmap(corr, cmap='YlGnBu', linewidths=.1, annot=True, square=False, ax=ax)
    ax.set_title(coin+' Global ' + method.capitalize() + ' Correlation', size=15)


def rolling_corr_heatmap(df, custom_params, ax):
    corrs = []
    coin = set_coin_name(df)
    target = custom_params['target']
    cols = custom_params['cols']
    window = custom_params['window']
    transformation = custom_params['transformation']
    periods_range = custom_params['periods_range']
    for periods in periods_range:
        rolling_corr = create_rolling_corr(df, target, cols, transformation, window, periods)
        corrs.append(list(rolling_corr.abs().mean()))
    corrs_df = pd.DataFrame(corrs, columns=cols, index=periods_range)
    sns.heatmap(corrs_df, cmap='YlGnBu', linewidths=.1, annot=True, square=False, ax=ax)
    ax.set_ylabel('Periods')
    ax.set_title(coin+' Rolling '+str(window)+' Window: '+transformation+' - Mean Absolute Correlation: ' + target, size=15)


def rolling_corr_plot(df, custom_params, start, renderer):
    df_out = df.copy()
    coin = set_coin_name(df)
    x_domain = set_x_domain(df)
    x_filter = set_x_filter(df, start)
    col = [custom_params['col']]
    transformation = custom_params['transformation']
    
    rolling_corr = create_rolling_corr(df, custom_params['target'], col, transformation, custom_params['window'], custom_params['periods'])
    new_col = set_transformation_cols(col, transformation)[0]
    label = new_col + ' Rolling Correlation'
    
    params = dict(vertical_spacing=0.1, row_heights=[1, .5], subplot_titles=(None, label), 
                  x_domain=x_domain, showlegend=True,
                  title_text=coin+' Buy-Hold-Sell Points with Rolling Correlation',
                  hlines=custom_params['hlines'], fig_size=(800, 600), xrange=x_filter, renderer=renderer)

    traces = [dict(mode='lines', x=df.index, y=df['close'], name='close', 
                   line={'color': 'CornFlowerBlue'}, showlegend=True, row=1),
              dict(mode='markers', x=df.index, y=df['close'], name='color', 
                   marker={'size': 4}, marker_color=df['color'], showlegend=False, row=1),
              dict(mode='lines', x=rolling_corr.index, y=rolling_corr[new_col], name=new_col,
                   line={'dash': 'solid'}, showlegend=True, row=2) 
             ]

    stacked_slider(traces, params)


def plot_results(predicted_data, true_data):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()


def plot_results_multiple(predicted_data, true_data, sequence_length, metric, threshold, title):
    fig, ax = plt.subplots(figsize=(10,5))
    seq_metrics = evaluate_sequence_predictions(true_data, predicted_data)
    if metric == 'kendalltau':
        taus = [x[0] for x in seq_metrics['kendalltau']]
        vs = [x[1] for x in seq_metrics['kendalltau']]
    else:
        vs = seq_metrics[metric]
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, a in enumerate(predicted_data):
        padding = [None for p in range(i * sequence_length)]
        if vs[i] <= threshold:
            color = 'green'
        else:
            color = 'red'
        ax.plot(padding + a, label='Prediction', color=color)
    ax.set_title(title)
    plt.show()


def normalization_hist_plot(p1, p2, title):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
    p1[0].hist(ax=ax1)
    ax1.set_title(p1[1])
    p2[0].hist(ax=ax2)
    ax2.set_title(p2[1])
    plt.suptitle(title)
    plt.show()