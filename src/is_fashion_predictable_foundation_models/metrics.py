import numpy as np
import pandas as pd


def mase(y_true: pd.DataFrame, y_pred: pd.DataFrame, seasonality: int) -> np.ndarray:
    """
    NOTE: y_true must contain at least `seasonality` time steps before the first time index
        of y_pred, plus the number of time steps in y_pred.
    """
    time_split = y_pred.index[0]
    index_y_pred = y_true.index.get_loc(time_split)
    assert index_y_pred > seasonality + 1, (
        f"To compute the MASE, you need {seasonality} time steps of history"
        f" before timestamp {time_split}."
    )

    numerator = np.mean(
        np.abs(y_true.values[index_y_pred:index_y_pred + len(y_pred)] - y_pred.values), axis=0
    )
    gt = y_true.values[seasonality:index_y_pred - 1]
    gt_shifted = y_true.values[:index_y_pred - 1 - seasonality]
    denominator = np.mean(np.abs(gt - gt_shifted), axis=0)
    return numerator / denominator


def mape(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> np.ndarray:
    time_split = y_pred.index[0]
    index_y_pred = y_true.index.get_loc(time_split)

    numerator = y_true.values[index_y_pred:index_y_pred + len(y_pred)] - y_pred.values
    denominator = y_true.values[index_y_pred:index_y_pred + len(y_pred)]
    mape = np.abs(numerator / denominator)
    mape[mape == np.inf] = np.nan
    return np.nanmean(mape, axis=0)


def smape(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> np.ndarray:
    time_split = y_pred.index[0]
    index_y_pred = y_true.index.get_loc(time_split)

    numerator = np.abs(y_true.values[index_y_pred:index_y_pred + len(y_pred)] - y_pred.values)
    denominator = 0.5 * (
        np.abs(y_true.values[index_y_pred:index_y_pred + len(y_pred)]) + np.abs(y_pred.values)
    )
    smape = np.abs(numerator / denominator)
    smape[smape == np.inf] = np.nan
    return np.nanmean(smape, axis=0)
