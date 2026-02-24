import numpy as np
import pandas as pd

def walk_forward_splits(dates: pd.Series, n_splits: int = 5, test_size_days: int = 63):
    """
    dates: série de datas (datetime), alinhada às linhas de X/y
    Faz splits por blocos de tempo.
    test_size_days ~ 63 (3 meses úteis), ajuste depois.
    """
    unique_dates = pd.Index(pd.to_datetime(dates).sort_values().unique())
    if len(unique_dates) < (n_splits * test_size_days + 50):
        # fallback: reduz splits automaticamente
        n_splits = max(1, len(unique_dates) // test_size_days - 1)

    splits = []
    for i in range(n_splits):
        test_end = len(unique_dates) - i * test_size_days
        test_start = test_end - test_size_days
        if test_start <= 0:
            break

        test_dates = set(unique_dates[test_start:test_end])
        train_dates = set(unique_dates[:test_start])

        splits.append((train_dates, test_dates))

    splits = list(reversed(splits))
    return splits

def dates_to_indices(dates: pd.Series, train_dates: set, test_dates: set):
    d = pd.to_datetime(dates)
    train_idx = np.flatnonzero(d.isin(train_dates))
    test_idx = np.flatnonzero(d.isin(test_dates))
    return train_idx, test_idx
