import os
from typing import Dict

import pandas as pd
from config import WEEKS_IN_YEAR
from metrics import mape, mase, smape


def load_time_series_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.freq = df.index.inferred_freq
    return df


def save_forecast(models_forecasts: Dict[str, pd.DataFrame], forecasts_dir: str) -> None:
    if not os.path.exists(forecasts_dir):
        os.mkdir(forecasts_dir)
    for model_name, model_predictions in models_forecasts.items():
        model_predictions.to_csv(os.path.join(forecasts_dir, f"{model_name}.csv"))


def load_forecast(forecasts_dir: str) -> Dict[str, pd.DataFrame]:
    return {
        single_model_forecast_path.replace('.csv', ''):
        pd.read_csv(os.path.join(forecasts_dir, single_model_forecast_path), index_col=0)
        for single_model_forecast_path in os.listdir(forecasts_dir)
    }


def eval_predictions(ts_signal: pd.DataFrame,
                     predictions: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    error_metrics = {}
    for model_name, model_prediction in predictions.items():
        error_metrics[model_name] = pd.DataFrame(
            columns=["mase", "mape", "smape"], index=ts_signal.columns
        )
        error_metrics[model_name]["mase"] = mase(
            ts_signal, model_prediction, seasonality=WEEKS_IN_YEAR
        ).round(4)
        error_metrics[model_name]["mape"] = mape(ts_signal, model_prediction).round(4)
        error_metrics[model_name]["smape"] = smape(ts_signal, model_prediction).round(4)
    return error_metrics


def mean_errors_ranking(error_metrics: Dict[str, pd.DataFrame], output_path: str) -> None:
    mean_result = pd.DataFrame(
        {model_name: model_metrics.mean() for model_name, model_metrics in error_metrics.items()}
    ).T.sort_values(by="mase", ascending=False)
    mean_result.to_csv(output_path)