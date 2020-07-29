import pandas as pd


# TIME DATA

def extract_timestamp(x: str, dt_format: str):
    return pd.to_datetime(x, format=dt_format)


def get_date_attributes(df, dt_column):
    df['year'] = df[dt_column].apply(lambda s: s.year)
    df['month'] = df[dt_column].apply(lambda s: s.month)
    df['day'] = df[dt_column].apply(lambda s: s.day)
    df['hour'] = df[dt_column].apply(lambda s: s.hour)
    df['weekday'] = df[dt_column].apply(lambda s: s.dayofweek)
    return df


def create_rolling_average_col(df, col_to_average: str, time_window: str):
    # time_window: e.g '4h'
    df[col_to_average + '_rolling' + '_' + time_window] = pd.DataFrame(df[col_to_average].rolling(time_window).mean())
    return df


# MULTILEVEL CATEGORICAL DATA

def get_value_counts_of_df(df):
    counts_dict = dict()
    for col in df.columns:
        counts_dict[col] = len(df[col].value_counts())
    return counts_dict


def delete_columns_with_high_value_counts(df, exclude_cols, max_value_threshold):
    keep = []
    for idx, i in enumerate(df.columns):
        if i not in exclude_cols:
            num_values = len(df[i].value_counts())
            if num_values < max_value_threshold:
                keep.append(i)
    return df[keep]


def create_other_category(df, col_name, threshold=.05):
    frequencies = df[col_name].value_counts(normalize=True)
    small_categories = frequencies[frequencies < threshold].index
    df[col_name] = df[col_name].replace(small_categories, "Other")
    return df
