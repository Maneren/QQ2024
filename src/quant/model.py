from __future__ import annotations

from pathlib import Path
from time import sleep
import time
from typing import TYPE_CHECKING, cast

import keras
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from keras import optimizers, regularizers
from keras.layers import Dense, Dropout, InputLayer
from keras.models import Sequential
from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from quant.data import Data, TeamData
from quant.ranking import Elo, EloByLocation
from quant.types import IModel, Match, Opp, match_to_opp

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
        self.old_matches: pd.DataFrame = pd.DataFrame()
        self.old_outcomes: pd.Series = pd.Series()
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

    RANKING_COLUMNS: tuple[str, ...] = (
        "HomeElo",
        "AwayElo",
        "EloByLocation",
    )
    MATCH_PARAMETERS = len(TeamData.COLUMNS) + len(RANKING_COLUMNS)
    TRAINING_DATA_COLUMNS: tuple[str, ...] = (*RANKING_COLUMNS, *TeamData.MATCH_COLUMNS)

    def create_dataframe(self, active_matches: pd.DataFrame) -> pd.DataFrame:
        """Get matches to predict outcome for."""
        return cast(
            pd.DataFrame,
            active_matches.apply(
                lambda x: self.get_match_parameters(match_to_opp(Match(0, *x))),
                axis=1,
            ),
        )

    def get_match_parameters(self, match: Opp) -> pd.Series:
        """Get parameters for given match."""
        home_elo = self.elo.team_rating(match.HID)
        away_elo = self.elo.team_rating(match.AID)
        elo_by_location_prediction = self.elo_by_location.predict(match)

        rankings = pd.Series(
            [
                home_elo,
                away_elo,
                elo_by_location_prediction,
            ],
            index=self.RANKING_COLUMNS,
        )

        data_parameters = self.data.get_match_parameters(match)

        return pd.concat([rankings, data_parameters], axis=0)

    def train_ai(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""
        training_data = []
        outcomes_list = []

        for match in (Match(*x) for x in dataframe.itertuples()):
            match_parameters = self.get_match_parameters(match_to_opp(match))

            training_data.append(match_parameters)
            outcomes_list.append(match.H)

            self.data.add_match(match)
            self.elo.add_match(match)
            self.elo_by_location.add_match(match)

        training_dataframe = pd.DataFrame(
            training_data, columns=pd.Index(self.TRAINING_DATA_COLUMNS)
        )

        outcomes = pd.Series(outcomes_list)

        self.old_matches = training_dataframe
        self.old_outcomes = outcomes

        self.ai.train(training_dataframe, outcomes)
        self.trained = True

    def train_ai_reg(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""

        if Path("data/training_dataframe.csv").is_file():
            print("Loading training dataframe...")
            training_dataframe = pd.read_csv("data/training_dataframe.csv", index_col=0)
            outcomes = pd.read_csv("data/outcomes.csv").iloc[:, 1]

        else:
            print("Creating training dataframe...")
            training_data = []
            outcomes_list = []

            for match in (Match(*x) for x in dataframe.itertuples()):
                match_parameters = self.get_match_parameters(match_to_opp(match))

                training_data.append(match_parameters)
                outcomes_list.append(match.HSC - match.ASC)

                self.data.add_match(match)
                self.elo.add_match(match)
                self.elo_by_location.add_match(match)

            training_dataframe = pd.DataFrame(
                training_data, columns=pd.Index(self.TRAINING_DATA_COLUMNS)
            )

            outcomes = pd.Series(outcomes_list, name="Outcome")

            training_dataframe.to_csv("data/training_dataframe.csv")
            outcomes.to_csv("data/outcomes.csv")

        self.old_matches = training_dataframe
        self.old_outcomes = outcomes

        self.ai.train_reg(training_dataframe, outcomes)
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


def accuracy(y_true, y_pred):
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    return np.mean(sign_true == sign_pred)


class PlotLosses(Callback):
    def __init__(self, x_train, y_train, x_val, y_val):
        super(PlotLosses, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy_train = []
        self.accuracy_val = []

        # Enable interactive mode for live updates
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()

    def on_epoch_end(self, epoch, logs={}):
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.i += 1

        start = time.time()
        accuracy_train = accuracy(self.y_train, self.model.predict(self.x_train))
        accuracy_val = accuracy(self.y_val, self.model.predict(self.x_val))
        elapsed = time.time() - start
        print(
            f"Accuracy train: {accuracy_train:.4f} val: {accuracy_val:.4f} in {elapsed:.2f}s"
        )
        self.accuracy_train.append(accuracy_train)
        self.accuracy_val.append(accuracy_val)

        # Clear the current axes and redraw
        self.ax.clear()
        self.ax.set_title("Training and Validation Loss/Accuracy")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss/Accuracy")

        self.ax.plot(self.x, self.losses, label="loss")
        self.ax.plot(self.x, self.val_losses, label="val_loss")

        self.ax.plot(self.x, self.accuracy_train, label="accuracy_train")
        self.ax.plot(self.x, self.accuracy_val, label="accuracy_val")
        self.ax.legend()

        # Update the plot
        plt.locator_params(axis="both", nbins=10)
        plt.draw()
        plt.pause(0.01)  # Small pause to allow plot to update

    def on_train_end(self, logs={}):
        # Turn off interactive mode and show final plot
        plt.ioff()
        plt.show()


class ActivationLogger(Callback):
    def __init__(self, layer_name, data):
        super().__init__()
        self.layer_name = layer_name
        self.data = data
        self.epoch_activations = []

    def on_epoch_end(self, epoch, logs=None):
        layer_output = keras.Model(
            inputs=self.model.layers[0].input,
            outputs=self.model.get_layer(self.layer_name).output,
        )
        activation_values = layer_output(self.data, training=False)
        self.epoch_activations.append(activation_values.numpy())


class FeatureSensitivityLogger(Callback):
    def __init__(self, data):
        super().__init__()
        self.mean_absolute_sensitivity_over_epochs = []
        self.mean_sensitivity_over_epochs = []
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        x_tensor = tf.convert_to_tensor(self.data, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predictions = self.model(x_tensor)
        sensitivities = tape.gradient(predictions, x_tensor)

        all_sensitivities = sensitivities.numpy()

        mean_absolute_sensitivity = np.mean(np.abs(all_sensitivities), axis=0)
        self.mean_absolute_sensitivity_over_epochs.append(mean_absolute_sensitivity)

        mean_sensitivity = np.mean(all_sensitivities, axis=0)
        self.mean_sensitivity_over_epochs.append(mean_sensitivity)


class Ai:
    """Class for training and predicting."""

    model: keras.Model
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    EARLY_STOPPING = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )

    def __init__(self):
        """Create a new Model from a XGBClassifier."""
        self.initialized = False

    def train(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Return trained model."""
        if not self.initialized:
            self.model = xgb.XGBClassifier()
            self.initialized = True

        self.model = self.model.fit(training_dataframe, outcomes)

    def train_reg(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Return trained model."""

        print("Training model...")
        print("Data shape:", training_dataframe.shape)
        print(training_dataframe.describe())
        print("Outcomes shape:", outcomes.shape)
        print(outcomes.describe())

        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            training_dataframe,
            outcomes,
            test_size=0.3,
            shuffle=True,
            random_state=42,
        )

        if not self.initialized:
            tf.random.set_seed(42)
            self.initialize_model()
            self.initialized = True
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            x_train = self.x_scaler.fit_transform(x_train)
            y_train = self.y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
        else:
            x_train = self.x_scaler.transform(x_train)
            y_train = self.y_scaler.transform(y_train.to_numpy().reshape(-1, 1))

        x_val = self.x_scaler.transform(x_val)
        y_val = self.y_scaler.transform(y_val.to_numpy().reshape(-1, 1))

        plot_losses = PlotLosses(x_train, y_train, x_val, y_val)
        activation_logger = ActivationLogger(layer_name="dense", data=x_train)
        feature_sensitivity_logger = FeatureSensitivityLogger(data=x_train)

        print("Fitting model...")
        print("x_train shape:", x_train.shape)
        print(pd.DataFrame(x_train).describe())
        print("y_train shape:", y_train.shape)
        print(pd.DataFrame(y_train).describe())
        input("Press enter to continue...")

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
                self.EARLY_STOPPING,
                activation_logger,
                feature_sensitivity_logger,
            ],
        )

        print("MAE:", metrics.mean_absolute_error(y_val, self.model.predict(x_val)))
        print("MSE:", metrics.mean_squared_error(y_val, self.model.predict(x_val)))
        outcomes_sign = np.sign(y_train)
        predictions_sign = np.sign(self.model.predict(x_train))
        print(
            "Accuracy (train):",
            np.sum(outcomes_sign == predictions_sign) / len(outcomes_sign),
        )
        outcomes_sign = np.sign(y_val)
        predictions_sign = np.sign(self.model.predict(x_val))
        print(
            "Accuracy (val):",
            np.sum(outcomes_sign == predictions_sign) / len(outcomes_sign),
        )

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

        # Plot sensitivities over epochs
        plt.figure(figsize=(10, 6))
        for i, feature_name in enumerate(training_dataframe.columns.tolist()):
            plt.plot(
                range(1, len(ma_sensitivity_over_epochs) + 1),
                ma_sensitivity_over_epochs[:, i],
                label=feature_name,
            )

        plt.title("Mean absolute Feature Sensitivities Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Absolute Sensitivity")
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
        plt.grid(True)
        plt.show()

    def initialize_model(self) -> None:
        """Initialize model from path."""
        self.model = Sequential()

        l2 = regularizers.L2(0.01)

        self.model.add(InputLayer(shape=(29,)))
        # self.model.add(
        #     Dense(10, activation="tanh", use_bias=True, kernel_regularizer=l2)
        # )
        self.model.add(
            Dense(32, activation="tanh", use_bias=True, kernel_regularizer=l2)
        )
        # self.model.add(Dropout(0.05))
        # self.model.add(
        #     Dense(64, activation="tanh", use_bias=True, kernel_regularizer=l2)
        # )
        self.model.add(Dense(1, use_bias=True))

        # Adam
        optimizer = optimizers.Adam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=True
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
