# Onde colocar ftime?
# NÃO USAR SIMILARIDADE POR COSSENO
# TODO:
# import warnings

# with warnings.catch_warnings(record=True) as w:
#     warnings.simplefilter("always")

#     # código que pode gerar warnings
#     foo()

#     for warn in w:
#         my_logger.log_warning(str(warn.message))

from abc import ABC, abstractmethod
from collections import defaultdict
import os
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Any, Literal, Callable
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


from Classes.time_series import TimeSeries


class TrainLogger:
    def __init__(self, is_active: bool):
        self.is_active = is_active
        self.identation = 0

    def push_identation(self):
        self.identation += 1
    
    def pop_identation(self):
        self.identation -= 1

    def log(self, msg):
        self._write(f"[LOG] {msg}")
    
    def warning(self, msg):
        self._write(f"[WARNING] {msg}")
    
    def error(self, msg):
        self._write(f"[ERROR] {msg}")

    def _write(self, msg):
        if self.is_active:
            print(f"{'\t' * self.identation}{msg}")


class TrainStopWatcher:
    def __init__(self):
        self.elapsed = 0
    
    def start(self):
        self._started = time.perf_counter()

    def stop(self):
        self.elapsed = time.perf_counter() - self._started

class TrainResult:
    def __init__(self, fit_time):
        self.fit_time = fit_time


class ModelWrapper(ABC):
    def __init__(self, name: str, is_online: bool, requires_sequence: bool, enabled: bool = True):
        self.name = name
        self.is_online = is_online
        self.enabled = enabled
        self.requires_sequence = requires_sequence

    @abstractmethod
    def fit(self, X, y, logger: TrainLogger) -> TrainResult:
        pass
    
    @abstractmethod
    def predict_in_sample(self, X, logger: TrainLogger):
        pass

    @abstractmethod
    def predict_osa(self, X, logger: TrainLogger):
        pass


class SklearnWrapper(ModelWrapper):
    def __init__(
        self,
        name: str,
        is_online: bool,
        enabled: bool,
        estimator: BaseEstimator,
        param_grid: dict[str, Any],
        scoring = "neg_mean_squared_error",
        n_jobs = -1
    ):
        super().__init__(name, is_online, False, enabled)
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs

    def fit(self, X, y, logger: TrainLogger) -> TrainResult:
        if not isinstance(self.estimator, BaseEstimator):
            raise Exception("Cannot fit self.estimator because it's not a sklearn.base.BaseEstimator")
        
        sw = TrainStopWatcher()

        gs = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            error_score='raise'
        )

        sw.start()
        gs.fit(X, y)
        sw.stop()

        logger.log(f"BestEstimator:")
        logger.push_identation()
        logger.log(f"BestParams: {gs.best_params_}")
        logger.log(f"BestScore: {gs.best_score_}")
        logger.pop_identation()

        self._best_estimator = gs.best_estimator_

        return TrainResult(sw.elapsed)

    def predict_in_sample(self, X, logger):
        if self._best_estimator is None:
            raise Exception("You must fit before predict.")
    
        return self._best_estimator.predict(X)

    def predict_osa(self, X, logger):
        if self._best_estimator is None:
            raise Exception("You must fit before predict.")
    
        return self._best_estimator.predict(X)


def build_rnn_model(
    meta,
    cell_type="lstm",
    units=64,
    dropout=0.2,
    lr=0.001
):
    input_shape = meta["X_shape_"][1:]

    model = Sequential()
    model.add(Input(shape=input_shape))

    if cell_type == "lstm":
        model.add(LSTM(units))
    else:
        model.add(GRU(units))

    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )

    return model


class SciKerasWrapper(ModelWrapper):
    def __init__(
        self,
        name: str,
        is_online: bool,
        enabled: bool,
        model_builder: Callable[..., Any],
        param_grid: dict[str, Any],
        scoring="neg_mean_squared_error",
        n_jobs=1,
        epochs=20,
        batch_size=32,
        verbose=0,
    ):
        super().__init__(name, is_online, True, enabled)

        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self._best_estimator = None

        self.estimator = KerasRegressor(
            model=model_builder,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

    def fit(self, X, y, logger: TrainLogger) -> TrainResult:
        sw = TrainStopWatcher()

        gs = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            error_score="raise"
        )

        sw.start()
        gs.fit(X, y)
        sw.stop()

        logger.log("BestEstimator:")
        logger.push_identation()
        logger.log(f"BestParams: {gs.best_params_}")
        logger.log(f"BestScore: {gs.best_score_}")
        logger.pop_identation()

        self._best_estimator = gs.best_estimator_

        return TrainResult(sw.elapsed)

    def predict_in_sample(self, X, logger):
        if self._best_estimator is None:
            raise Exception("You must fit before predict.")

        return self._best_estimator.predict(X)

    def predict_osa(self, X, logger):
        if self._best_estimator is None:
            raise Exception("You must fit before predict.")

        return self._best_estimator.predict(X)


class SequenceFeatureBuilder:
    """
    Unified feature builder for:
      - sklearn classical models (2D)
      - LSTM / GRU SciKeras models (3D)
    """

    def __init__(self, window_size: int, sequence_mode: bool = False):
        self.window_size = window_size
        self.sequence_mode = sequence_mode
        self.scaler = StandardScaler()

    def create_lag_matrix(self, series: pd.Series) -> pd.DataFrame:
        data = {
            f"lag_{k}": series.shift(k)
            for k in range(1, self.window_size + 1)
        }
        return pd.DataFrame(data)

    def fit_transform(self, series: pd.Series):
        X = self.create_lag_matrix(series).dropna()
        X_scaled = self.scaler.fit_transform(X)

        if self.sequence_mode:
            X_scaled = X_scaled.reshape(
                X_scaled.shape[0],
                X_scaled.shape[1],
                1
            )

        y = series.loc[X.index]
        return X.index, X_scaled, y

    def transform(self, series: pd.Series, valid_index):
        X_all = self.create_lag_matrix(series)
        X = X_all.loc[valid_index].dropna()
        X_scaled = self.scaler.transform(X)

        if self.sequence_mode:
            X_scaled = X_scaled.reshape(
                X_scaled.shape[0],
                X_scaled.shape[1],
                1
            )

        return X.index, X_scaled


class Trainer:
    @staticmethod
    def train(
        models: list[ModelWrapper],
        rw_series: dict[str, TimeSeries],
        target_series: dict[str, TimeSeries],
        train_size: float = 0.6,
        output_dir: str = "results"
    ):
        RW_RESIDUALS_COLUMN = "rw_residuals"
        RW_SERIES_COLUMN = "rw"
        TARGET_COLUMN = "y"

        def split_train_val(series: pd.DataFrame, train_size: float):
            if not (0 < train_size < 1):
                raise Exception("train_size must be between 0 and 1.")

            n_train = int(train_size * len(series))
            return series[:n_train], series[n_train:]

        def build_rows(ts_name, index, target, pred, model, phase):
            ae = np.abs(target - pred)
            se = (target - pred) ** 2
            ape = np.where(target != 0, np.abs((target - pred) / target), np.nan)

            return pd.DataFrame({
                "series_name": ts_name,
                "index": index,
                "target": target,
                "pred": pred,
                "model": model.name,
                "AE": ae,
                "SE": se,
                "TUE": np.nan,
                "APE": ape,
                "Paradigm": "online" if model.is_online else "offline",
                "Phase": phase
            })

        logger = TrainLogger(is_active=True)
        results = defaultdict(list)

        logger.log("Starting training pipeline...")

        for ts_name, rw in rw_series.items():
            if ts_name not in target_series:
                logger.warning(f"'{ts_name}' missing in target_series, skipping...")
                continue

            logger.log(f"Series: {ts_name}")

            tval = rw["train"].dropna()
            train, val = split_train_val(tval, train_size)

            window_size = max(1, int(0.1 * len(train)))

            for model in models:
                logger.push_identation()
                if not model.enabled:
                    logger.log(f"Skipping {model.name}...")
                    logger.pop_identation()
                    continue

                logger.log(f"Training {model.name}...")

                # Detect if model requires sequence input
                sequence_mode = getattr(model, "requires_sequence", False)

                feature_builder = SequenceFeatureBuilder(
                    window_size=window_size,
                    sequence_mode=sequence_mode
                )

                # TRAIN FEATURES
                train_idx, X_train, y_train = feature_builder.fit_transform(
                    train[RW_RESIDUALS_COLUMN]
                )

                # VAL FEATURES
                val_idx, X_val = feature_builder.transform(
                    tval[RW_RESIDUALS_COLUMN],
                    val.index
                )

                y_val = val[RW_RESIDUALS_COLUMN].loc[val_idx]

                # Original target
                ts_original = target_series[ts_name]["train"][TARGET_COLUMN]
                target_train = ts_original.loc[train_idx]
                target_val = ts_original.loc[val_idx]

                # RW predictions
                rw_train_pred = train[RW_SERIES_COLUMN].loc[train_idx]
                rw_val_pred = val[RW_SERIES_COLUMN].loc[val_idx]

                # FIT
                logger.push_identation()
                train_result = model.fit(X_train, y_train, logger)
                logger.log(f"FitTime: {train_result.fit_time}")
                logger.pop_identation()

                # PREDICT TRAIN
                logger.log("Predicting in-sample...")
                in_sample_model_preds = model.predict_in_sample(X_train, logger)

                # PREDICT VAL
                logger.log("Predicting OSA...")
                osa_model_preds = model.predict_osa(X_val, logger)

                # Residual correction
                in_sample_preds = in_sample_model_preds + rw_train_pred.values
                osa_preds = osa_model_preds + rw_val_pred.values

                # Build outputs
                df_train = build_rows(
                    ts_name,
                    target_train.index,
                    target_train.values,
                    in_sample_preds,
                    model,
                    "train"
                )

                df_val = build_rows(
                    ts_name,
                    target_val.index,
                    target_val.values,
                    osa_preds,
                    model,
                    "val"
                )

                results[model.name].append(df_train)
                results[model.name].append(df_val)

                logger.log(f"Completed {model.name} in {ts_name}")
                logger.pop_identation()

        # SAVE RESULTS
        logger.log("Saving results...")
        os.makedirs(output_dir, exist_ok=True)

        for model_name, dfs in results.items():
            df_model = pd.concat(dfs, ignore_index=True)
            path = os.path.join(output_dir, f"{model_name}.csv")
            df_model.to_csv(path, index=False)

        logger.log(f"Results saved in '{output_dir}'")