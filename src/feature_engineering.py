def create_features(df):
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['rolling_mean_7'] = df['sales'].rolling(7).mean()
    df['rolling_std_7'] = df['sales'].rolling(7).std()
    return df.dropna()
