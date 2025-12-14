from typing import Protocol
from quant.types import Match, Opp


class RankingModel(Protocol):
    """Ranking model interface."""

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the model."""
        raise NotImplementedError


class TeamElo:
    """
    Class for keeping track of a team's Elo.

    Attributes:
        opponents: Sum of Elo ratings of opponents
        games: Number of games
        wins: Number of wins
        rating: Current Elo rating

    """

    A: int = 400
    K: int = 40
    BASE: int = 160
    opponents: int
    games: int
    wins: int
    rating: float

    def __init__(self) -> None:
        """Initialize TeamElo."""
        self.games = 0
        self.wins = 0
        self.rating = 1000

    def adjust(self, opponent_elo: float, win: int) -> None:
        """
        Adjust Elo rating based on one match.

        Args:
            opponent_elo: Elo rating of the other team
            win: 1 for win, 0 for loss

        """
        self.games += 1
        self.wins += 1 if win else 0

        expected = self.predict(opponent_elo)

        self.rating += self.K * (win - expected)

    def predict(self, opponent_elo: float) -> float:
        """
        Predict outcome of a match with opponent of given Elo.

        Args:
            opponent_elo: Elo of the opponent

        Returns:
            Probability of winning (0..1)

        """
        d = opponent_elo - self.rating
        return 1 / (1 + self.BASE ** (d / self.A))

    def __str__(self) -> str:
        """Create a string representation of the team's Elo."""
        return (
            f"{self.rating:>4} ({self.games:>4}, "
            f"{self.wins:>4}, {self.wins / self.games * 100:>6.2f}%)"
        )


class Elo(RankingModel):
    """Class for the Elo ranking model."""

    teams: dict[int, TeamElo]

    def __init__(self) -> None:
        """Initialize Elo model."""
        self.teams = {}

    def __str__(self) -> str:
        """Create a string representation of the model."""
        return "Team  Elo Opponents Games  Wins  WinRate\n" + "\n".join(
            f" {team:>2}: {elo}"
            for team, elo in sorted(
                self.teams.items(), key=lambda item: -item[1].rating
            )
        )

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        home_team = self.teams.setdefault(match.HID, TeamElo())
        away_team = self.teams.setdefault(match.AID, TeamElo())

        home_elo = home_team.rating
        away_elo = away_team.rating

        home_team.adjust(away_elo, match.H)
        away_team.adjust(home_elo, match.A)

    def rankings(self) -> dict[int, float]:
        """Return normalized rankings."""
        max_elo = max(elo.rating for elo in self.teams.values())
        return {team: teamElo.rating / max_elo for team, teamElo in self.teams.items()}

    def team_rating(self, team_id: int) -> float:
        """Return Elo rating of a team."""
        return self.teams.setdefault(team_id, TeamElo()).rating

    def reset(self) -> None:
        """Reset the model."""
        self.teams = {}


class EloByLocation(RankingModel):
    """Class for the Elo ranking model."""

    teams_home: dict[int, TeamElo]
    teams_away: dict[int, TeamElo]

    def __init__(self) -> None:
        """Initialize Elo model."""
        self.teams_home = {}
        self.teams_away = {}

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        home_elo = self.teams_home.setdefault(match.HID, TeamElo())
        away_elo = self.teams_away.setdefault(match.AID, TeamElo())

        # memorize elo values before they change
        home_elo_value = home_elo.rating
        away_elo_value = away_elo.rating

        home_elo.adjust(away_elo_value, match.H)
        away_elo.adjust(home_elo_value, match.A)

    def predict(self, match: Opp) -> float | None:
        """
        Predicts how the match might go.

        Float from 0 to 1 = chance of H to win
        None means no data
        """
        home_elo = self.teams_home.setdefault(match.HID, TeamElo())
        away_elo = self.teams_away.setdefault(match.AID, TeamElo())

        played_enough = home_elo.games >= 10 and away_elo.games >= 10
        return 100 * home_elo.predict(away_elo.rating) if played_enough else 400

    def reset(self) -> None:
        """Reset the model."""
        self.teams_home.clear()
        self.teams_away.clear()
