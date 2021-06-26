import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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