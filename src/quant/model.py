from __future__ import annotations

from itertools import product
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
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
    FeatureSensitivityLogger,
    PlotLosses,
    accuracy,
    weighted_accuracy,
)


tf.random.set_seed(42)

SCALAR_FEATURES: tuple[str, ...] = (
    "HE",
    "AE",
    "HEBL",
    "AEBL",
)
MATCH_SCALAR_FEATURES: tuple[str, ...] = (
    *SCALAR_FEATURES,
    *TeamData.MATCH_SCALAR_FEATURES,
)
MATCH_VECTOR_FEATURES = TeamData.MATCH_VECTOR_FEATURES
TRAINING_DATA_FEATURES: tuple[str, ...] = (
    *MATCH_SCALAR_FEATURES,
    *TeamData.MATCH_VECTOR_FEATURES,
)


def initialize_model(
    nd_shape: tuple[int, ...], scalar_shape: tuple[int, ...]
) -> tf.Model:
    """Initialize the model."""
    print("Initializing model...")
    main_input = Input(shape=nd_shape, name="matrix_input")
    scalar_input = Input(shape=scalar_shape, name="vector_input")

    x = DepthwiseConv1D(
        24,
        strides=4,
        activation="leaky_relu",
        padding="same",
        depthwise_regularizer=L2(),
        data_format="channels_first",
    )(main_input)
    x = MaxPooling1D(4, strides=2, data_format="channels_first")(x)

    y = DepthwiseConv1D(
        12,
        strides=4,
        activation="leaky_relu",
        padding="same",
        depthwise_regularizer=L2(),
        data_format="channels_first",
    )(main_input)
    y = MaxPooling1D(4, strides=2, data_format="channels_first")(y)

    z = DepthwiseConv1D(
        6,
        strides=4,
        activation="leaky_relu",
        padding="same",
        depthwise_regularizer=L2(),
        data_format="channels_first",
    )(main_input)
    z = MaxPooling1D(4, strides=2, data_format="channels_first")(z)

    combined = Concatenate()([Flatten()(x), Flatten()(y), Flatten()(z), scalar_input])
    combined = Dropout(0.2)(combined)
    combined = Dense(
        8,
        activation="leaky_relu",
        kernel_regularizer=L2(),
        use_bias=True,
        bias_regularizer=L2(),
    )(combined)
    output = Dense(1, name="output", use_bias=True)(combined)

    model = keras.Model(inputs=[main_input, scalar_input], outputs=output)

    print(model.summary())

    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.01))

    return model


class Model:
    """Class for training and predicting."""

    model: keras.Model
    x_nd_scaler: StandardScaler
    x_scalar_scaler: StandardScaler
    y_scaler: StandardScaler

    def __init__(self, nd_shape: tuple[int, ...], scalar_shape: tuple[int, ...]):
        self.initialized = False
        self.model = initialize_model(nd_shape, scalar_shape)
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

        split_result = model_selection.train_test_split(
            nd_data,
            scalar_data,
            outcomes[:, np.newaxis],
            test_size=0.3,
            shuffle=True,
            random_state=42,
        )

        x_nd_train = np.asarray(split_result[0])
        x_nd_val = np.asarray(split_result[1])
        x_scalar_train = np.asarray(split_result[2])
        x_scalar_val = np.asarray(split_result[3])
        y_train = np.asarray(split_result[4])
        y_val = np.asarray(split_result[5])

        flat_x_train = x_nd_train.reshape(x_nd_train.shape[0], -1)
        flat_x_val = x_nd_val.reshape(x_nd_val.shape[0], -1)

        if not self.initialized:
            self.initialized = True

            print("Fitting scaler to data...")
            self.x_nd_scaler.fit(flat_x_train)
            self.x_scalar_scaler.fit(x_scalar_train)
            self.y_scaler.fit(y_train)

        print("Normalizing data...")
        norm_x_nd_train = np.asarray(self.x_nd_scaler.transform(flat_x_train))
        x_scalar_train = np.asarray(self.x_scalar_scaler.transform(x_scalar_train))
        y_train = np.asarray(self.y_scaler.transform(y_train))

        norm_x_nd_val: np.ndarray = np.asarray(self.x_nd_scaler.transform(flat_x_val))
        x_scalar_val = np.asarray(self.x_scalar_scaler.transform(x_scalar_val))

        original_nd_train_shape = np.asarray(split_result[0]).shape
        original_nd_val_shape = np.asarray(split_result[1]).shape
        x_nd_train = norm_x_nd_train.reshape(original_nd_train_shape)
        x_nd_val = norm_x_nd_val.reshape(original_nd_val_shape)

        y_val = np.asarray(self.y_scaler.transform(y_val))

        x_train = (x_nd_train, x_scalar_train)
        x_val = (x_nd_val, x_scalar_val)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
        plot_losses = PlotLosses(x_train, y_train, x_val, y_val)
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
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks=[
                plot_losses,
                early_stopping,
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
                    TRAINING_DATA_FEATURES,
                    map(float, ma_sensitivity_over_epochs[-1]),
                ),
                key=lambda item: item[1],
                reverse=True,
            ),
            sep="\n",
        )

    def save_model(self, path: str | Path) -> None:
        """Save ML model."""
        self.model.save_model(path)
