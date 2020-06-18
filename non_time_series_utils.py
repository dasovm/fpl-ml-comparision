import numpy as np
import pandas as pd

y_col = "total_points"
point_separator = 4


def generate_df():
    df = pd.DataFrame(np.random.randint(0, 10, size=(100, 5)),
                      columns=[y_col] + list('abcd'))
    return df


def split_df_to_train_test(df, split_rate=0.8):
    test = df.sample(frac=(1-split_rate), random_state=1)
    train = df.drop(test.index)

    return train, test


def split_df_to_X_y(df):
    df = df.copy()
    df_y = df.pop(y_col)
    return df.values, df_y.values


def get_classes_from_y(y):
    return (y > point_separator).astype(int)
