from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn import metrics, model_selection

from quant.player import Player
from quant.ranking import Elo, EloByLocation
from quant.data import Data, TeamData
from quant.types import Match, Opp, Summary, match_to_opp

if TYPE_CHECKING:
    import os

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def month_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return 12 * (a.year - b.year) + (a.month - b.month)


class Model:
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
        summ: pd.DataFrame,
        opps: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        games_increment = inc[0]
        # summary = Summary(*summ.iloc[0])

        for row in games_increment.itertuples():
            self.data.add_match(Match(*row))

        return pd.DataFrame()

        # if not self.trained:
        #     train_size = self.TRAIN_SIZE * self.FIRST_TRAIN_MOD
        #     print(
        #         f"Initial training on {games_increment[-train_size:].shape[0]}"
        #         f" matches with bankroll {summary.Bankroll}"
        #     )
        #     self.train_ai_reg(cast(pd.DataFrame, games_increment[-train_size:]))
        #     self.last_retrain = summary.Date
        # elif games_increment.shape[0] > 0:
        #     increment_season = int(games_increment.iloc[0]["Season"])
        #     if self.season_number != increment_season:
        #         self.elo.reset()
        #         self.elo_by_location.reset()
        #         self.season_number = increment_season
        #
        #     self.old_matches = pd.concat(
        #         [
        #             self.old_matches.iloc[-self.TRAIN_SIZE :],
        #             self.create_dataframe(games_increment),
        #         ],
        #     )
        #
        #     self.old_outcomes = cast(
        #         pd.Series,
        #         pd.concat(
        #             [
        #                 self.old_outcomes.iloc[-self.TRAIN_SIZE :],
        #                 games_increment.HSC - games_increment.ASC,
        #             ],
        #         ),
        #     )
        #
        #     if month_diff(summary.Date, self.last_retrain) >= 2:
        #         print(
        #             f"{summary.Date}: retraining on {self.old_matches.shape[0]}"
        #             f" matches with bankroll {summary.Bankroll}"
        #         )
        #         self.ai.train_reg(self.old_matches, self.old_outcomes)
        #         self.last_retrain = summary.Date
        #         self.budget = summary.Bankroll
        #
        #     self.update_models(games_increment)
        #
        # active_matches = cast(pd.DataFrame, opps[opps["Date"] == summary.Date])
        #
        # if active_matches.shape[0] == 0 or summary.Bankroll < (self.budget * 0.9):
        #     return pd.DataFrame(
        #         data=0,
        #         index=np.arange(active_matches.shape[0]),
        #         columns=pd.Index(["BetH", "BetA"], dtype="str"),
        #     )
        #
        # dataframe = self.create_dataframe(active_matches)
        # probabilities = self.ai.get_probabilities_reg(dataframe)
        # bets = self.player.get_betting_strategy(probabilities, active_matches, summary)
        #
        # new_bets = pd.DataFrame(
        #     data=bets,
        #     columns=pd.Index(["BetH", "BetA"], dtype="str"),
        #     index=active_matches.index,
        # )
        #
        # return new_bets.reindex(opps.index, fill_value=0)

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

        outcomes = pd.Series(outcomes_list)

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


class Ai:
    """Class for training and predicting."""

    model: tensorflow.keras.Model

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
        if not self.initialized:
            self.model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=10)
            self.initialized = True

        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            training_dataframe.to_numpy(),
            outcomes.to_numpy(),
            test_size=0.3,
            random_state=2,
            shuffle=True,
        )
        self.model.fit(x_train, y_train)
        print("MAE:", metrics.mean_absolute_error(y_val, self.model.predict(x_val)))
        print(
            *sorted(
                zip(training_dataframe.columns, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            ),
            sep="\n",
        )

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
