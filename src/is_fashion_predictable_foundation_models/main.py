import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from model.foundation_model import (
    ChronosForecastModel, FlowStateForecastModel, FoundationForecastModelType, MoiraiForecastModel,
    TimesfmForecastModel, TirexForecastModel
)
from utils import (
    eval_predictions, load_forecast, load_time_series_csv, mean_errors_ranking, save_forecast
)

# Load the fashion time series data
fashion_ts = load_time_series_csv("data/us_female_sneakers.csv")
fashion_ts_train = fashion_ts.loc[:"2023-01-01"
                                 ]  # Split the dataset into training data up to January 1, 2023
compute_prediction = False  # Set this parameter to True if you want to re-compute the prediction.

# Define foundations models
foundation_models = {
    FoundationForecastModelType.chronos.value: ChronosForecastModel(device_map='cpu'),
    FoundationForecastModelType.timesfm.value: TimesfmForecastModel(),
    FoundationForecastModelType.moirai.value: MoiraiForecastModel(),
    FoundationForecastModelType.tirex.value: TirexForecastModel(),
    FoundationForecastModelType.flowstate.value: FlowStateForecastModel(device='cpu')
}

if compute_prediction:
    # Generate forecasts from both statistical and deep learning models
    models_forecasts = {
        model_name: model.predict(fashion_ts_train)
        for model_name, model in foundation_models.items()
    }
    # save forecasts in directory model_forecasts/
    save_forecast(models_forecasts=models_forecasts, forecasts_dir="model_forecasts")
else:
    # Load models' forecasts from .csv stored in model_forecasts/ directory.
    models_forecasts = load_forecast(forecasts_dir="model_forecasts")

# Evaluate the forecasts using the metrics MASE, SMAPE and MAPE
error_metrics = eval_predictions(fashion_ts, models_forecasts)

# Rank the models based on mean error metrics and save the results to a CSV file
mean_errors_ranking(error_metrics=error_metrics, output_path="mean_errors_ranking.csv")