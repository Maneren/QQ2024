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
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    InputLayer,
    MaxPooling1D,
)
from keras.models import Sequential
from keras.regularizers import L2
from numpy.typing import NDArray
from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler

from quant.data import Data, TeamData
from quant.ranking import Elo, EloByLocation
from quant.types import IModel, Match, Opp, Team, match_to_opp
from quant.utils import (
    ActivationLogger,
    FeatureSensitivityLogger,
    PlotLosses,
    accuracy,
    weighted_accuracy,
)

if TYPE_CHECKING:
    import os

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def month_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return 12 * (a.year - b.year) + (a.month - b.month)


class Model(IModel):
    """Main class."""

    TRAIN_SIZE: int = 2000
    FIRST_TRAIN_MOD: int = 10

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.elo = Elo()
        self.elo_by_location = EloByLocation()
        # self.player = Player()
        self.ai = Ai()
        self.trained = False
        self.data = Data()
        self.season_number: int = 0
        # self.budget: int = 0
        self.old_matches = np.ndarray((0, 0, 0))
        self.old_outcomes = np.ndarray((0,))
        # self.last_retrain: pd.Timestamp = pd.Timestamp(0)

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for match in (Match(*row) for row in games_increment.itertuples()):
            self.elo.add_match(match)
            self.elo_by_location.add_match(match)
            self.data.add_match(match)

    def place_bets(
        self,
        _summ: pd.DataFrame,
        _opps: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        games_increment, _players_increment = inc

        if not self.trained:
            self.train_ai_reg(games_increment)

        raise StopIteration

        return pd.DataFrame()

    SCALAR_COLUMNS: tuple[str, ...] = (
        "HE",
        "AE",
        "HEBL",
        "AEBL",
        "N",
        "POFF",
    )
    MATCH_SCALAR_COLUMNS: tuple[str, ...] = (
        *SCALAR_COLUMNS,
        *TeamData.MATCH_SCALAR_COLUMNS,
    )
    TRAINING_DATA_COLUMNS: tuple[str, ...] = (
        *MATCH_SCALAR_COLUMNS,
        *TeamData.MATCH_VECTOR_COLUMNS,
    )

    def get_match_parameters(
        self, match: Opp
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get parameters for given match."""
        home_elo = self.elo.team_rating(match.HID)
        away_elo = self.elo.team_rating(match.AID)
        home_elo_by_location = self.elo_by_location.team_rating(match.HID, Team.Home)
        away_elo_by_location = self.elo_by_location.team_rating(match.AID, Team.Away)

        vector_parameters, scalar_parameters = self.data.get_match_parameters(match)

        scalar_parameters = np.concatenate(
            [
                [
                    home_elo,
                    away_elo,
                    home_elo_by_location,
                    away_elo_by_location,
                    match.N,
                    match.POFF,
                ],
                scalar_parameters,
            ],
            axis=None,
            dtype=np.float64,
        )

        return vector_parameters, scalar_parameters

    def train_ai_reg(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""

        if Path("data/training.npz").is_file():
            print("Loading training dataframe...")

            loaded = np.load("data/training.npz")
            nd_data = loaded["nd_data"]
            scalar_data = loaded["scalar_data"]
            outcomes = loaded["outcomes"]

        else:
            print("Creating training dataframe...")
            samples = len(dataframe)
            nd_data = np.zeros(
                (samples, len(TeamData.MATCH_VECTOR_COLUMNS), TeamData.N)
            )
            scalar_data = np.zeros((samples, len(self.MATCH_SCALAR_COLUMNS)))
            outcomes = np.zeros((samples,))

            for i, match in enumerate(starmap(Match, dataframe.itertuples())):
                nd, scalar = self.get_match_parameters(match_to_opp(match))

                if i % 1000 == 0:
                    print(f"Processed {i}/{len(dataframe)} matches...")

                nd_data[i] = nd
                scalar_data[i] = scalar
                outcomes[i] = np.float64(match.HSC - match.ASC)

                self.data.add_match(match)
                self.elo.add_match(match)
                self.elo_by_location.add_match(match)

            np.savez(
                "data/training.npz",
                nd_data=nd_data,
                scalar_data=scalar_data,
                outcomes=outcomes,
            )

        self.old_matches = nd_data
        self.old_outcomes = outcomes

        self.ai.train_reg(nd_data, scalar_data, outcomes)
        self.trained = True


def calculate_elo_accuracy(data: list[list[int]]) -> float:
    """Calculate the accuracy of ELO predictions."""
    correct_predictions = 0
    total_games = len(data)
    games = np.array(data)[:, :-1]
    outcomes = np.array(data)[:, -1].clip(0, 1).round(decimals=0)
    for i in range(len(data)):
        elo_home = games[i][0]
        elo_away = games[i][1]
        outcome = outcomes[i]

        # Predict home win if home ELO is greater than away ELO
        predicted_outcome = 1 if elo_home > elo_away else 0

        # Compare predicted outcome with actual outcome
        if predicted_outcome == outcome:
            correct_predictions += 1

    # Calculate accuracy as a percentage
    return correct_predictions / total_games


class Ai:
    """Class for training and predicting."""

    model: keras.Model
    x_nd_scaler: StandardScaler
    x_scalar_scaler: StandardScaler
    y_scaler: StandardScaler

    def __init__(self):
        """Create a new Model from a XGBClassifier."""
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

            norm_x_nd_train: np.ndarray = self.x_nd_scaler.fit_transform(flat_x_train)
            x_scalar_train = self.x_scalar_scaler.fit_transform(x_scalar_train)
            y_train = self.y_scaler.fit_transform(y_train)
        else:
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
            epochs=3,
            batch_size=200,
            verbose=1,
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

        # all_epoch_activations = np.array(activation_logger.epoch_activations)
        #
        # print(np.shape(all_epoch_activations))
        # transposed_activations = np.transpose(all_epoch_activations, (2, 0, 1))
        #
        # # Reshape to neurons x (samples * epochs)
        # flattened_activations = transposed_activations.reshape(
        #     transposed_activations.shape[0], -1
        # )
        #
        # print(np.shape(flattened_activations))
        # correlation_matrix = np.corrcoef(
        #     flattened_activations
        # )  # Transpose to get neurons x epochs in case of averaging
        #
        # # Visualize the correlation matrix
        # plt.figure(figsize=(20, 16))
        # sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        # plt.title("Neuron Activation Correlation")
        # plt.xlabel("Neurons")
        # plt.ylabel("Neurons")
        # plt.show()
        #
        # transposed_activations = np.transpose(all_epoch_activations[-1])
        # np.shape(transposed_activations)
        #
        # plt.figure(figsize=(12, 8))
        # sns.heatmap(
        #     transposed_activations,
        #     cmap="viridis",
        #     annot=False,
        #     xticklabels=False,
        #     yticklabels=False,
        # )
        # plt.title("Neuron Activations for All Samples (Last Epoch)")
        # plt.xlabel("Samples")
        # plt.ylabel("Neurons")
        # plt.show()

        ma_sensitivity_over_epochs = np.array(
            feature_sensitivity_logger.mean_absolute_sensitivity_over_epochs
        )

        print(ma_sensitivity_over_epochs.shape)

        print(
            *sorted(
                zip(
                    Model.TRAINING_DATA_COLUMNS,
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
            Model.TRAINING_DATA_COLUMNS, ma_sensitivity_over_epochs.T
        ):
            plot.plot(epochs, values, label=feature_name)

        def on_plot_hover(event):
            # Iterating over each data member plotted
            for curve in plot.get_lines():
                # Searching which data member corresponds to current mouse position
                if curve.contains(event)[0]:
                    print(f"over {curve.get_label()}")

        plt.title("Mean absolute Feature Sensitivities Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Absolute Sensitivity")
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
        plt.grid(True)
        # fig.canvas.mpl_connect("motion_notify_event", on_plot_hover)
        plt.show()

    def initialize_model(
        self, nd_shape: tuple[int, ...], scalar_shape: tuple[int, ...]
    ) -> None:
        """Initialize model from path."""
        main_input = Input(shape=nd_shape[1:], name="main_input")
        scalar_input = Input(shape=scalar_shape[1:], name="scalar_input")

        x = Conv1D(
            16, 6, activation="relu", padding="same", kernel_regularizer=L2(0.01)
        )(main_input)
        x = Dropout(0.05)(x)
        x = MaxPooling1D(8, strides=2)(x)
        x = Flatten()(x)

        # Mix in the scalar features
        combined = Concatenate()([x, scalar_input])

        dense = Dense(16, activation="tanh", kernel_regularizer=L2(0.005))(combined)
        output = Dense(1, name="output", use_bias=True)(dense)

        self.model = keras.Model(inputs=[main_input, scalar_input], outputs=output)

        print(self.model.summary())

        # Adam
        optimizer = optimizers.Adam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False
        )

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
