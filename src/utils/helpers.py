import copy
import datetime as dt
import numpy as np
import scipy.stats as stats
import pandas as pd


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


def set_train_end(df, test_periods):
    return df.shape[0] - test_periods


def set_windows(nrows, window):
    windows = list(range(0, nrows, window))
    rem = nrows % window
    # If there is a remainder and it is less than 50% of the window add to the final window
    if rem > 0 and rem < (window * 0.50):
        last_el = windows.pop(-1)
        windows.append(last_el + rem)
    elif rem > 0:
        windows.append(windows[-1] + rem)
    else:
        windows.append(windows[-1] + window)
        
    return windows


# Preprocessing using input scaler object
def data_preprocessing_scaling(scaler, df, cols, test_periods):
    # Select columns and remove blanks
    df_out = df[cols].dropna().copy()
    # Identify end time period and create training data
    train_end = set_train_end(df_out, test_periods)
    train_df = df_out[:train_end]
    scaler.fit(train_df)
    # Transform training data
    train_df_scaled = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns, index=train_df.index)
    # Create test and transform test data
    test_df = df_out[train_end:]
    test_df_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)
    scaler_out = copy.deepcopy(scaler)
    return df_out, train_df_scaled, test_df_scaled, scaler_out


def standardize_window(scaler, df, window):
    nrows = df.shape[0]
    windows = set_windows(nrows, window)
    df_normalized = df.copy()
    scalers = []
    for i in range(len(windows)-1):
        window_start = windows[i]
        window_end = windows[i+1]
        scaler.fit(df_normalized[window_start:window_end])
        df_normalized[window_start:window_end] = scaler.transform(df_normalized[window_start:window_end])
        # Make a deep copy of the scaler fitted for the window to use later for inverse transformation of the window
        scaler_window = copy.deepcopy(scaler)
        scalers.append([(window_start, window_end), scaler_window])
    return df_normalized, scalers


def data_preprocessing_standardize(scaler, df, cols, test_periods, window):
    # Select columns and remove blanks
    df_out = df[cols].dropna().copy()
    nrows = df_out.shape[0]
    # Identify end time period and create training data
    train_end = set_train_end(df_out, test_periods)
    # If no window then call data_preprocessing_scaling to perform global normalization on training data
    if window == -1: 
        return data_preprocessing_scaling(scaler, df_out, cols, test_periods)
    elif window > 0:
        train_df = df_out[:train_end]
        train_df_scaled, train_scalers = standardize_window(scaler, train_df, window)
        test_df = df_out[train_end:]
        test_df_scaled, test_scalers = standardize_window(scaler, test_df, window)
        return df_out, train_df_scaled, test_df_scaled, (train_scalers, test_scalers)


def inverse_transform_global(scaler, data):
    df_transformed = pd.DataFrame(scaler.inverse_transform(data), columns=data.columns, index=data.index)
    return df_transformed


def inverse_transform_window(scalers, data):
    df_transformed = data.copy()
    for window, scaler in scalers:
        window_start = window[0]
        window_end = window[1]
        df_transformed[window_start:window_end] = scaler.inverse_transform(df_transformed[window_start:window_end])
    return df_transformed
    

def data_preprocessing_pct_change(df, cols, test_periods, window):
    # Select columns and remove blanks
    df_out = df[cols].dropna().copy()
    nrows = df_out.shape[0]
    # Identify end time period and create training data
    train_end = set_train_end(df_out, test_periods)
    if window == -1:
        # In this case the validation data would only have visibility to 1 data point in training data
        # Therefore, I made the decision just to calculation pct_change across entire df then split
        df_transformed = df_out.pct_change().dropna()
    elif window > 0:
        df_transformed = df_out.copy()
        windows = set_windows(df_transformed.shape[0], window)
        for i in range(len(windows)-1):
            window_start = windows[i]
            window_end = windows[i+1]
            for col in df_transformed.columns:
                df_transformed[col][window_start:window_end] = \
                df_transformed[col][window_start:window_end] / df_transformed[col][window_start:window_end].iat[0] - 1
    # Identify end time period and create training data
    train_end = set_train_end(df_transformed, test_periods)
    train_df_scaled = df_transformed[:train_end]
    # Create test and transform test data
    test_df_scaled = df_transformed[train_end:]
    return df_out, train_df_scaled, test_df_scaled


def create_timeseries_data(df, target_col, sequence_length):
    n = df.shape[0]
    d = df.shape[1]
    y = df[target_col][sequence_length:].values
    if d == 1:
        data = df.values.reshape(-1, 1)
        X = np.hstack(tuple([data[i: n-j, :] for i, j in enumerate(range(sequence_length, 0, -1))]))
        X = X.reshape(-1, sequence_length, 1)
    elif d > 1:
        X = np.stack([df[i: j] for i, j in enumerate(range(sequence_length, n))], axis=0)
    return X, y


def update_model_input_dim(config, X_shape):
    # Make a deep copy of configs to send out for custom params for each model
    config_out = copy.deepcopy(config)
    config_out['model']['layers'][0]['input_timesteps'] = X_shape[1]
    config_out['model']['layers'][0]['input_dim'] = X_shape[2]
    return config_out


def prediction_error(true_data, predicted_data):
    return true_data - predicted_data


def mda(true_data, predicted_data):
    return np.mean((np.sign(true_data[1:] - true_data[:-1]) == np.sign(predicted_data[1:] - predicted_data[:-1])).astype(int))


def mae(true_data, predicted_data):
    return np.mean(np.abs(prediction_error(true_data, predicted_data)))


def mase(true_data, predicted_data, shift):
    return mae(true_data, predicted_data) / mae(true_data[shift:], true_data[:-shift])


def accuracy(true_data, predicted_data):
    return np.mean(true_data == predicted_data)


def set_sequence_trend(a, threshold=0.05):
    a_pct = (a[1:] - a[0]) / np.abs(a[0])
    a_pct_end = a_pct[-1]
    a_pct_mean = np.mean(a_pct)
    a_pct_max = np.max(a_pct)
    a_pct_min = np.min(a_pct)
    trend = 0
    if a_pct_max > threshold and a_pct_end > 0 and a_pct_mean > 0:
        trend = 2
    elif a_pct_max < threshold and a_pct_end > 0 and a_pct_mean > 0:
        trend = 1
    elif a_pct_min < -threshold and a_pct_end < 0 and a_pct_mean < 0:
        trend = -2
    elif a_pct_min > -threshold and a_pct_end < 0 and a_pct_mean < 0:
        trend = -1
    return trend


def evaluation_metrics(true_data, predicted_data):
    metrics = {}
    # Flatten predictions
    predicted_data_flat = np.array(predicted_data).flatten()
    # Limit true_data to length of predictions
    true_data_flat = np.array(true_data[:predicted_data_flat.shape[0]]).flatten()
    # Mean Directional Accuracy
    metrics['mda'] = mda(true_data_flat, predicted_data_flat)
    # Mean Absolute Error
    metrics['mae'] = mae(true_data_flat, predicted_data_flat)
    # Mean Absolute Scaled Error
    metrics['mase'] = mase(true_data_flat, predicted_data_flat, 1)
    # Kendall’s tau correlation measure
    tau, p_value = stats.kendalltau(true_data_flat, predicted_data_flat)
    metrics['kendalltau'] = (tau, p_value)
    # Trend accuracy custom metric
    true_trend = set_sequence_trend(true_data_flat)
    predicted_trend = set_sequence_trend(predicted_data_flat)
    metrics['accuracy'] = (true_trend, predicted_trend, accuracy(true_trend, predicted_trend))
    return metrics


def mean_metric_results(results, metric):
    return np.mean(results[metric])


def mean_metric_seq_results(results, metric):
    return np.mean(np.array([d[metric] for d in results]).flatten())


def mean_tau_stat_sig(results, alpha):
    return np.mean(np.array([d[1] for d in results['kendalltau']]) <= alpha)


def mean_tau_seq_stat_sig(results, alpha):
    mean_ps = []
    seq_taus = [d['kendalltau'] for d in results]
    for seq in seq_taus:
        ps = np.array([tau[1] for tau in seq])
        mean_ps.append(np.mean(ps <= alpha))
    return np.mean(mean_ps)


def mean_accuracy(results):
    return np.mean(np.array([d[2] for d in results['accuracy']]))


def mean_accuracy_seq(results):
    mean_accs = []
    seq_accs = [d['accuracy'] for d in results]
    for seq in seq_accs:
        accs = np.array([acc[2] for acc in seq])
        mean_accs.append(np.mean(accs))
    return np.mean(mean_accs)


def set_accuracy_trend_color(actual_trend, predicted_trend):
    if actual_trend == predicted_trend:
        color = 'green'
    elif actual_trend > 0 and predicted_trend > 0 or actual_trend < 0 and predicted_trend < 0:
        color = 'orange'
    else:
        color = 'red'
    return color


def evaluate_sequence_predictions(true_data, predicted_data):
    seq_metrics = {}
    ##### Make this dynamic for all potential metrics
    mdas = []
    maes = []
    mases = []
    taus = []
    acc = []
    for i in range(len(predicted_data)):
        predicted = np.array(predicted_data[i])
        l = len(predicted)
        actual = np.array(true_data[l*i:l*(i+1)].flatten(), dtype='float32')
        metrics = evaluation_metrics(actual, predicted)
        mdas.append(metrics['mda'])
        maes.append(metrics['mae'])
        mases.append(metrics['mase'])
        taus.append(metrics['kendalltau'])
        acc.append(metrics['accuracy'])
    seq_metrics['mda'] = mdas
    seq_metrics['mae'] = maes
    seq_metrics['mase'] = mases
    seq_metrics['kendalltau'] = taus
    seq_metrics['accuracy'] = acc
    return seq_metrics


def evaluate_model(model, X, y, sequence_length, prediction_len):
    # Make predictions
    predictions = model.predict_sequences_multiple(X, sequence_length, prediction_len)
    seq_metrics = evaluate_sequence_predictions(y, predictions)
    return predictions, seq_metrics


