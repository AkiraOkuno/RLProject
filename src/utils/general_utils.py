import datetime
import pickle


def cumulative_distinct_values(series):
    cumsum = (~series.dropna().duplicated()).cumsum()
    series.loc[cumsum.index] = cumsum
    series = series.fillna(method="ffill")
    return series


def last_day_of_month(d: datetime.date) -> datetime.date:

    date = datetime.date(d.year + d.month // 12, d.month % 12 + 1, 1) - datetime.timedelta(days=1)
    return datetime.datetime(date.year, date.month, date.day)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def open_pickle(path):
    with open(path, "rb") as f:
        output = pickle.load(f)
    return output


def normalize(df, cols):
    df_normalized = df.copy()
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df_normalized[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df_normalized
