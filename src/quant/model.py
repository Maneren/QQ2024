from __future__ import annotations

from itertools import product, starmap
from pathlib import Path
from typing import TYPE_CHECKING, cast

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import (
    Concatenate,
    Dense,
    DepthwiseConv1D,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from keras.regularizers import L2
from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler

from quant.data import TeamData
from quant.utils import (
    ActivationLogger,
    FeatureSensitivityLogger,
    PlotLosses,
    accuracy,
    weighted_accuracy,
)

if TYPE_CHECKING:
    import os

SCALAR_COLUMNS: tuple[str, ...] = (
    "HE",
    "AE",
    "HEBL",
    "AEBL",
)
MATCH_SCALAR_COLUMNS: tuple[str, ...] = (
    *SCALAR_COLUMNS,
    *TeamData.MATCH_SCALAR_COLUMNS,
)
MATCH_VECTOR_COLUMNS = TeamData.MATCH_VECTOR_COLUMNS
TRAINING_DATA_COLUMNS: tuple[str, ...] = (
    *MATCH_SCALAR_COLUMNS,
    *TeamData.MATCH_VECTOR_COLUMNS,
)


class Model:
    """Class for training and predicting."""

    model: keras.Model
    x_nd_scaler: StandardScaler
    x_scalar_scaler: StandardScaler
    y_scaler: StandardScaler

    def __init__(self):
        self.initialized = False
        self.x_nd_scaler = StandardScaler()
        self.x_scalar_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def train_reg(
        self, nd_data: np.ndarray, scalar_data: np.ndarray, outcomes: np.ndarray
    ) -> None:
        """Return trained model."""

        print("Training model...")
        print("Data shape:", nd_data.shape)
        print("Outcomes shape:", outcomes.shape)

        x_nd_train: np.ndarray
        x_nd_val: np.ndarray
        x_scalar_train: np.ndarray
        x_scalar_val: np.ndarray
        y_train: np.ndarray
        y_val: np.ndarray

        x_nd_train, x_nd_val, x_scalar_train, x_scalar_val, y_train, y_val = cast(
            tuple[np.ndarray, ...],
            model_selection.train_test_split(
                nd_data,
                scalar_data,
                outcomes[:, np.newaxis],
                test_size=0.3,
                shuffle=True,
                random_state=42,
            ),
        )

        flat_x_train = x_nd_train.reshape(x_nd_train.shape[0], -1)
        flat_x_val = x_nd_val.reshape(x_nd_val.shape[0], -1)

        if not self.initialized:
            tf.random.set_seed(42)
            self.initialize_model(x_nd_train.shape, scalar_data.shape)
            self.initialized = True

            print("Normalizing data...")
            norm_x_nd_train: np.ndarray = self.x_nd_scaler.fit_transform(flat_x_train)
            x_scalar_train = self.x_scalar_scaler.fit_transform(x_scalar_train)
            y_train = self.y_scaler.fit_transform(y_train)
        else:
            print("Normalizing data...")
            norm_x_nd_train: np.ndarray = self.x_nd_scaler.transform(flat_x_train)
            x_scalar_train = self.x_scalar_scaler.transform(x_scalar_train)
            y_train = self.y_scaler.transform(y_train)

        norm_x_nd_val: np.ndarray = self.x_nd_scaler.transform(flat_x_val)
        x_scalar_val = self.x_scalar_scaler.transform(x_scalar_val)
        x_nd_train = norm_x_nd_train.reshape(x_nd_train.shape)
        x_nd_val = norm_x_nd_val.reshape(x_nd_val.shape)

        y_val = self.y_scaler.transform(y_val)

        x_train = [x_nd_train, x_scalar_train]
        x_val = [x_nd_val, x_scalar_val]

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
        plot_losses = PlotLosses(x_train, y_train, x_val, y_val)
        activation_logger = ActivationLogger(x_train)
        feature_sensitivity_logger = FeatureSensitivityLogger(
            x_nd_train, x_scalar_train
        )

        print("Fitting model...")
        print("x_train shape:", x_nd_train.shape)
        print("y_train shape:", y_train.shape)

        self.model.fit(
            x_train,
            y_train,
            shuffle=True,
            epochs=1000,
            batch_size=500,
            verbose=0,
            validation_data=(x_val, y_val),
            callbacks=[
                plot_losses,
                early_stopping,
                activation_logger,
                feature_sensitivity_logger,
            ],
        )

        predictions_train = self.model.predict(x_train)
        predictions_val = self.model.predict(x_val)

        for (data_name, y, predictions), (name, f) in product(
            [("train", y_train, predictions_train), ("val", y_val, predictions_val)],
            [
                ("MAE", metrics.mean_absolute_error),
                ("MSE", metrics.mean_squared_error),
                ("Accuracy", accuracy),
                ("Accuracy weighted", weighted_accuracy),
            ],
        ):
            print(f"{name} ({data_name}): {f(y, predictions)}")

        ma_sensitivity_over_epochs = np.array(
            feature_sensitivity_logger.mean_absolute_sensitivity_over_epochs
        )

        print(
            *sorted(
                zip(
                    TRAINING_DATA_COLUMNS,
                    map(float, ma_sensitivity_over_epochs[-1]),
                ),
                key=lambda item: item[1],
                reverse=True,
            ),
            sep="\n",
        )

        # Plot sensitivities over epochs
        fig = plt.figure(figsize=(10, 6))
        plot = fig.add_subplot()
        epochs = list(range(1, len(ma_sensitivity_over_epochs) + 1))
        for feature_name, values in zip(
            TRAINING_DATA_COLUMNS, ma_sensitivity_over_epochs.T
        ):
            plot.plot(epochs, values, label=feature_name)

        plt.title("Mean Absolute Feature Sensitivity Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Absolute Sensitivity")
        plt.legend(loc="upper right", bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.show()

    def initialize_model(
        self, nd_shape: tuple[int, ...], scalar_shape: tuple[int, ...]
    ) -> None:
        """Initialize model from path."""
        print("Initializing model...")
        main_input = Input(shape=nd_shape[1:], name="matrix_input")
        scalar_input = Input(shape=scalar_shape[1:], name="vector_input")

        x = DepthwiseConv1D(
            24,
            strides=4,
            activation="relu",
            padding="same",
            depthwise_regularizer=L2(),
            data_format="channels_first",
        )(main_input)
        x = MaxPooling1D(4, strides=2, data_format="channels_first")(x)

        y = DepthwiseConv1D(
            12,
            strides=4,
            activation="relu",
            padding="same",
            depthwise_regularizer=L2(),
            data_format="channels_first",
        )(main_input)
        y = MaxPooling1D(4, strides=2, data_format="channels_first")(y)

        z = DepthwiseConv1D(
            6,
            strides=4,
            activation="relu",
            padding="same",
            depthwise_regularizer=L2(),
            data_format="channels_first",
        )(main_input)
        z = MaxPooling1D(4, strides=2, data_format="channels_first")(z)

        combined = Concatenate()(
            [Flatten()(x), Flatten()(y), Flatten()(z), scalar_input]
        )
        combined = Dropout(0.5)(combined)
        combined = Dense(
            16,
            activation="tanh",
            kernel_regularizer=L2(),
            use_bias=True,
            bias_regularizer=L2(),
        )(combined)
        output = Dense(1, name="output", use_bias=True)(combined)

        self.model = keras.Model(inputs=[main_input, scalar_input], outputs=output)

        print(self.model.summary())

        optimizer = optimizers.Adam(learning_rate=0.01)

        self.model.compile(loss="mean_squared_error", optimizer=optimizer)

    def get_probabilities(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        return self.model.predict_proba(dataframe.to_numpy())

    def get_probabilities_reg(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        predicted_score_differences = self.model.predict(dataframe)
        return self.calculate_probabilities(predicted_score_differences)

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        self.model.save_model(path)

    def home_team_win_probability(self, score_difference: float) -> float:
        """Calculate the probability of home team winning based on score difference."""
        slope = 0.8  # range optimal 0.1 to 1. liked 0.3 and 0.5 (maybe 1)
        return 1 / (1 + np.exp(-slope * score_difference))

    def calculate_probabilities(self, score_differences: np.ndarray) -> pd.DataFrame:
        """Calculate the probabilities of teams winning based on score differences."""
        probabilities = []

        for score_difference in score_differences:
            home_prob = self.home_team_win_probability(score_difference)
            away_prob = 1 - home_prob
            probabilities.append((home_prob, away_prob))

        return pd.DataFrame(probabilities, columns=pd.Index(["WinHome", "WinAway"]))
