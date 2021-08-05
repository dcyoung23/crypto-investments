import datetime as dt
import scipy.stats as stats

class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		#print('Time taken: %s' % (end_dt - self.start_dt))


def set_x_domain(df):
    return (df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d"))


def set_x_filter(df, start):
    return [start, df.index.max().strftime("%Y-%m-%d")]


def set_coin_name(df):
    return str(df['name'].unique()[0])


def set_transformation_cols(cols, transformation):
    return [col+'_'+transformation for col in cols]


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


def create_rolling_corr(df, target, cols, transformation, window, periods):
    df_out = apply_transformation(df, cols, transformation, periods)
    new_cols = set_transformation_cols(cols, transformation)
    return df_out[target].rolling(window).corr(df_out[new_cols])


def evaluate_sequence_predictions(predicted_data, true_data):
    taus = []
    ps = []
    for i in range(len(predicted_data)):
        a = predicted_data[i]
        l = len(a)
        b = true_data[l*i:l*(i+1)].flatten()
        tau, p_value = stats.kendalltau(a, b)
        taus.append(tau)
        ps.append(p_value)
    return taus, ps


