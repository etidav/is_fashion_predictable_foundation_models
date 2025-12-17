import abc
import copy
import json
import os
import random
import tempfile
from enum import Enum
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from config import PREDICTION_ONE_YEAR, WEEK_FREQUENCY_TIMEINDEX
from pydantic import BaseModel
from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
from tirex import load_model
from tsfm_public import FlowStateForPrediction
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_STEPS = 1000
FINETUNED_CKPT_NAME = "model"
DEFAULT_MAX_CONTEXT = 256
DEFAULT_MAX_HORIZON = 256
DEFAULT_CONTEXT_LENGTH = 1680
DEFAULT_DEVICE = "cuda"
DEFAULT_SCALE_FACTOR = 0.46  # see Flowstate paper
DEFAULT_CONTEXT_LENGTH_CHRONOS = 208
DEFAULT_EVAL_SIZE_CHRONOS = 10000
DEFAULT_WARMUP_RATIO = 0.0


class FoundationForecastModelType(str, Enum):
    chronos = "chronos"
    timesfm = "timesfm"
    moirai = "moirai"
    tirex = "tirex"
    flowstate = "flowstate"


class FoundationForecastModel(BaseModel, metaclass=abc.ABCMeta):
    """
        Abstract class for foundation models.
    """

    horizon: int = PREDICTION_ONE_YEAR
    model_type: FoundationForecastModelType = None
    model: Any = None
    model_args: List[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_args is None:
            self.model_args = []
        if self.model is None:
            self.init_model()

    @abc.abstractmethod
    def init_model(self):
        pass

    @abc.abstractmethod
    def save_model(self, path: str):
        pass

    def save_model_args(self, path: str):

        model_args = {arg_name: getattr(self, arg_name) for arg_name in self.model_args}
        model_args_path = os.path.join(path, "model_args.json")
        with open(model_args_path, "w") as f:
            json.dump(model_args, f, indent=4)

    def format_predictions(
        self, predictions: np.ndarray, last_historical_index: pd.Timestamp,
        prediction_columns: List[str]
    ) -> pd.DataFrame:
        predictions[predictions < 0] = 0.
        predictions = predictions.astype('float64')
        predictions_index = pd.date_range(
            start=last_historical_index, periods=self.horizon + 1, freq=WEEK_FREQUENCY_TIMEINDEX
        )[1:]
        return pd.DataFrame(
            predictions,
            columns=prediction_columns,
            index=predictions_index,
        )

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "FoundationForecastModel":

        model_args_path = os.path.join(path, "model_args.json")
        if os.path.isfile(model_args_path):
            with open(model_args_path) as f:
                model_args = json.load(f)
        else:
            model_args = {}

        if not os.path.isdir(path):
            raise ValueError(f"No model saved at following directory path : {path}")

        return cls(
            model_path=path,
            **model_args,
            **kwargs,
        )


class ChronosForecastModel(FoundationForecastModel):
    model_type: FoundationForecastModelType = FoundationForecastModelType.chronos
    model_path: str = "amazon/chronos-2"
    model: Chronos2Pipeline = None
    id_column: str = "unique_ds"
    timestamp_column: str = "ds"
    target: str = "y"
    prediction_column_name: str = "predictions"

    zero_shot: bool = None
    learning_rate: float = None
    batch_size: int = None
    num_steps: int = None
    device_map: str = None
    context_length: int = None
    warmup_ratio: float = None
    eval_size: int = None

    class Config:
        arbitrary_types_allowed = True

    def init_model(self):
        self.zero_shot = True if self.zero_shot is None else self.zero_shot
        self.learning_rate = DEFAULT_LEARNING_RATE if self.learning_rate is None else self.learning_rate
        self.batch_size = DEFAULT_BATCH_SIZE if self.batch_size is None else self.batch_size
        self.num_steps = DEFAULT_NUM_STEPS if self.num_steps is None else self.num_steps
        self.device_map = DEFAULT_DEVICE if self.device_map is None else self.device_map
        self.context_length = DEFAULT_CONTEXT_LENGTH_CHRONOS if self.context_length is None else self.context_length
        self.warmup_ratio = DEFAULT_WARMUP_RATIO if self.warmup_ratio is None else self.warmup_ratio
        self.eval_size = DEFAULT_EVAL_SIZE_CHRONOS if self.eval_size is None else self.eval_size
        self.model_args = [
            "zero_shot", "learning_rate", "batch_size", "num_steps", "device_map", "context_length",
            "warmup_ratio", "eval_size"
        ]

        self.model = BaseChronosPipeline.from_pretrained(
            self.model_path, device_map=self.device_map
        )

    def format_single_ts(self, ts: pd.Series, ts_index: int) -> pd.DataFrame:

        return pd.DataFrame(
            {
                self.id_column: ts_index,
                self.timestamp_column: pd.to_datetime(ts.index),
                self.target: ts.values,
            }
        )

    def format_multiple_ts(
        self, data: Union[pd.DataFrame, pd.Series], time_index: str = None
    ) -> pd.DataFrame:
        formated_train_data = []
        if time_index is not None:
            data = data.loc[:time_index]
        for i, col in enumerate(data.columns):
            formated_train_data.append(self.format_single_ts(data[col], i + 1))
        formated_train_data = pd.concat(formated_train_data, axis=0).reset_index(drop=True)
        return formated_train_data

    def fit(self, historical_data: pd.DataFrame, time_index: Optional[str] = None):

        if self.zero_shot:
            print(
                "Foundation model instantiated with zero_shot=True. The model will not be trained and will be saved as-is."
            )
            return

        dataset = self.format_multiple_ts(historical_data, time_index=time_index)
        train_inputs = []
        val_inputs = []
        eval_set = random.sample(
            range(1, historical_data.shape[1] + 1), self.eval_size
        )  #To speed up the training, we don't evaluate the model on all the dataset but only a subsample.
        for item_id, group in dataset.groupby(self.id_column):
            train_inputs.append({
                "target": group[self.target].values[:-self.horizon],
            })
            if item_id in eval_set:
                val_inputs.append(
                    {
                        "target": group[self.target].values[-self.horizon - self.context_length:],
                    }
                )

        # Can't avoid saving the model at the end of the training so we save it in a temp file.
        # If you want to save the model, use the save function.
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.model.fit(
                inputs=train_inputs,
                validation_inputs=val_inputs,
                prediction_length=self.horizon,
                num_steps=self.num_steps,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                context_length=self.context_length,
                warmup_ratio=self.warmup_ratio,
                output_dir=tmp_dir,
                finetuned_ckpt_name=FINETUNED_CKPT_NAME
            )
            self.model = BaseChronosPipeline.from_pretrained(
                os.path.join(tmp_dir, FINETUNED_CKPT_NAME), device_map=self.device_map
            )

    def predict(self, historical_data: pd.DataFrame, time_index: str = None) -> pd.DataFrame:

        if time_index is not None:
            historical_data = copy.deepcopy(historical_data)
            historical_data.date_split(time_index)
        dataset = self.format_multiple_ts(historical_data, time_index=time_index)
        y_pred_df = self.model.predict_df(
            dataset,
            prediction_length=self.horizon,
            quantile_levels=[],
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            target=self.target
        )
        predictions = y_pred_df[self.prediction_column_name].values.reshape(-1, self.horizon).T
        return self.format_predictions(
            predictions, historical_data.index[-1], historical_data.columns
        )

    def save_model(self, path: str):

        self.model.save_pretrained(save_directory=path)
        self.save_model_args(path)


class TimesfmForecastModel(FoundationForecastModel):
    model_type: FoundationForecastModelType = FoundationForecastModelType.timesfm
    model_path: str = "google/timesfm-2.5-200m-pytorch"
    model: TimesFM_2p5_200M_torch = None

    zero_shot: bool = None
    max_context: int = None
    max_horizon: int = None
    normalize_inputs: bool = None
    force_flip_invariance: bool = True
    infer_is_positive: bool = True

    class Config:
        arbitrary_types_allowed = True

    def init_model(self):
        self.zero_shot = True if self.zero_shot is None else self.zero_shot
        self.max_context = DEFAULT_MAX_CONTEXT if self.max_context is None else self.max_context
        self.max_horizon = DEFAULT_MAX_HORIZON if self.max_horizon is None else self.max_horizon
        self.normalize_inputs = True if self.normalize_inputs is None else self.normalize_inputs
        self.force_flip_invariance = True if self.force_flip_invariance is None else self.force_flip_invariance
        self.infer_is_positive = True if self.infer_is_positive is None else self.infer_is_positive

        self.model_args = [
            "zero_shot", "max_context", "max_horizon", "normalize_inputs", "force_flip_invariance",
            "infer_is_positive"
        ]

        self.model = TimesFM_2p5_200M_torch.from_pretrained(self.model_path, torch_compile=True)
        self.model.compile(
            ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=self.normalize_inputs,
                force_flip_invariance=self.force_flip_invariance,
                infer_is_positive=self.infer_is_positive,
            )
        )

    def predict(self, historical_data: pd.DataFrame, time_index: str = None) -> pd.DataFrame:

        if time_index is not None:
            historical_data = copy.deepcopy(historical_data)
            historical_data.date_split(time_index)

        y_pred_df, _ = self.model.forecast(horizon=self.horizon, inputs=historical_data.values.T)
        predictions = y_pred_df.T
        return self.format_predictions(
            predictions, historical_data.index[-1], historical_data.columns
        )

    def save_model(self, path: str):

        self.model.save_pretrained(save_directory=path)
        self.save_model_args(path)


class MoiraiForecastModel(FoundationForecastModel):
    model_type: FoundationForecastModelType = FoundationForecastModelType.moirai
    model_path: str = "Salesforce/moirai-2.0-R-small"
    model: Moirai2Forecast = None

    zero_shot: bool = True
    prediction_length: int = None
    context_length: int = None
    target_dim: int = None
    feat_dynamic_real_dim: int = None
    past_feat_dynamic_real_dim: int = None
    batch_size: int = None

    class Config:
        arbitrary_types_allowed = True

    def init_model(self):
        self.zero_shot = True if self.zero_shot is None else self.zero_shot
        self.context_length = DEFAULT_CONTEXT_LENGTH if self.context_length is None else self.context_length
        self.prediction_length = self.horizon if self.prediction_length is None else self.prediction_length
        self.target_dim = 1 if self.target_dim is None else self.target_dim
        self.feat_dynamic_real_dim = 0 if self.feat_dynamic_real_dim is None else self.feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = 0 if self.past_feat_dynamic_real_dim is None else self.past_feat_dynamic_real_dim
        self.batch_size = DEFAULT_BATCH_SIZE if self.batch_size is None else self.batch_size

        self.model_args = [
            "zero_shot", "context_length", "target_dim", "feat_dynamic_real_dim",
            "past_feat_dynamic_real_dim", "batch_size"
        ]

        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.model_path),
            prediction_length=self.horizon,
            context_length=self.context_length,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=self.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
        )

    def predict(self, historical_data: pd.DataFrame, time_index: str = None) -> pd.DataFrame:

        if time_index is not None:
            historical_data = copy.deepcopy(historical_data)
            historical_data.date_split(time_index)

        predictions = self.model.predict(historical_data.values.T
                                        )[:, self.model.module.quantile_levels.index(0.5), :
                                         ]  # Retrieve the median prediction.
        return self.format_predictions(
            predictions.T, historical_data.index[-1], historical_data.columns
        )

    def save_model(self, path: str):

        self.model.module.save_pretrained(save_directory=path)
        self.save_model_args(path)


class TirexForecastModel(FoundationForecastModel):
    model_type: FoundationForecastModelType = FoundationForecastModelType.tirex
    model_path: str = "NX-AI/TiRex"
    model: Any = None

    zero_shot: bool = None

    class Config:
        arbitrary_types_allowed = True

    def init_model(self):
        self.zero_shot = True if self.zero_shot is None else self.zero_shot
        self.model_args = ["zero_shot"]
        self.model = load_model(self.model_path)

    def predict(self, historical_data: pd.DataFrame, time_index: str = None) -> pd.DataFrame:

        if time_index is not None:
            historical_data = copy.deepcopy(historical_data)
            historical_data.date_split(time_index)

        _, predictions = self.model.forecast(
            context=historical_data.values.T, prediction_length=self.horizon
        )
        return self.format_predictions(
            predictions.numpy().T, historical_data.index[-1], historical_data.columns
        )

    def save_model(self, path: str):
        print(
            "Can't save/load a Tirex Model for now. Only the pretrained model 'NX-AI/TiRex' is available."
        )
        return

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "TirexForecastModel":

        print(
            "Can't load a Tirex Model for now. Only the pretrained model 'NX-AI/TiRex' is available and will be loaded."
        )
        return cls(**kwargs)


class FlowStateForecastModel(FoundationForecastModel):
    model_type: FoundationForecastModelType = FoundationForecastModelType.flowstate
    model_path: str = "ibm-granite/granite-timeseries-flowstate-r1"
    model: FlowStateForPrediction = None

    zero_shot: bool = None
    device: str = None
    scale_factor: float = None
    batch_size: int = None

    class Config:
        arbitrary_types_allowed = True

    def init_model(self):
        self.zero_shot = True if self.zero_shot is None else self.zero_shot
        self.device = DEFAULT_DEVICE if self.device is None else self.device
        self.scale_factor = DEFAULT_SCALE_FACTOR if self.scale_factor is None else self.scale_factor
        self.batch_size = DEFAULT_BATCH_SIZE if self.batch_size is None else self.batch_size

        self.model_args = ["zero_shot", "device", "scale_factor", "batch_size"]

        self.model = FlowStateForPrediction.from_pretrained(self.model_path).to(self.device)

    def predict(self, historical_data: pd.DataFrame, time_index: str = None) -> pd.DataFrame:

        if time_index is not None:
            historical_data = copy.deepcopy(historical_data)
            historical_data.date_split(time_index)

        values = historical_data.to_numpy().T[:, :, np.newaxis]
        past_values = torch.tensor(values, device=self.device, dtype=torch.float32)
        predictions = self.model(
            past_values=past_values,
            prediction_length=self.horizon,
            batch_first=True,
            scale_factor=self.scale_factor,
        )
        predictions = predictions.prediction_outputs[:, :, 0].cpu().detach().numpy(
        ).T  #Retrieve the median prediction.
        return self.format_predictions(
            predictions, historical_data.index[-1], historical_data.columns
        )

    def save_model(self, path: str):

        self.model.model.save_pretrained(save_directory=path)
        self.save_model_args(path)