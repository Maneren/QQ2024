from itertools import product, starmap
from operator import add
from typing import Iterable

from numpy.typing import NDArray


from quant.queue import RollingQueue
from quant.types import Match, Opp, Team, TeamID
import pandas as pd
import numpy as np


def prefix_columns(
    prefixes: Iterable[str], columns: tuple[str, ...]
) -> tuple[str, ...]:
    """Generate column names by combining prefixes with base columns."""
    return tuple(starmap(add, product(prefixes, columns)))


class TeamData:
    """Hold data of one team, both as home and away."""

    N = 32

    TEAM_VECTOR_FEATURES: tuple[str, ...] = (
        "WR",  # wins
        "WRH",  # wins home
        "WRA",  # wins away
        "PSA",  # points scored
        "PSAH",  # points scored home
        "PSAA",  # points scored away
        "FGM",  # successful throws
        "FGA",  # attempted throws
        "FG3M",  # successful 3-point throws
        "FG3A",  # attempted 3-point throws
        "FTM",  # successful free throws
        "FTA",  # attempted free throws
        "ORB",  # offensive rebounds
        "DRB",  # defensive rebounds
        "RB",  # rebounds
        "AST",  # assists
        "STL",  # steals
        "BLK",  # blocks
        "TOV",  # turnovers
        "PF",  # fouls
    )

    TEAM_SCALAR_FEATURES: tuple[str, ...] = ("DSLM",)  # days since last match

    MATCH_VECTOR_FEATURES: tuple[str, ...] = prefix_columns(
        ["H_", "A_"], TEAM_VECTOR_FEATURES
    )

    MATCH_SCALAR_FEATURES = prefix_columns(["H_", "A_"], TEAM_SCALAR_FEATURES)

    def __init__(
        self,
    ) -> None:
        """Init data structure."""
        self.date_last_match: pd.Timestamp = pd.to_datetime("1975-11-07")

        self.wins = RollingQueue(TeamData.N)
        self.wins_home = RollingQueue(TeamData.N)
        self.wins_away = RollingQueue(TeamData.N)

        self.scored = RollingQueue(TeamData.N)
        self.scored_home = RollingQueue(TeamData.N)
        self.scored_away = RollingQueue(TeamData.N)

        # Field goals made/attempted
        self.fgm = RollingQueue(TeamData.N)
        self.fga = RollingQueue(TeamData.N)

        # 3-point field goals made/attempted
        self.fg3m = RollingQueue(TeamData.N)
        self.fg3a = RollingQueue(TeamData.N)

        # Free throws made/attempted
        self.ftm = RollingQueue(TeamData.N)
        self.fta = RollingQueue(TeamData.N)

        # Rebounds
        self.orb = RollingQueue(TeamData.N)
        self.drb = RollingQueue(TeamData.N)
        self.rb = RollingQueue(TeamData.N)

        # Other stats
        self.ast = RollingQueue(TeamData.N)
        self.stl = RollingQueue(TeamData.N)
        self.blk = RollingQueue(TeamData.N)
        self.tov = RollingQueue(TeamData.N)
        self.pf = RollingQueue(TeamData.N)

    def _get_days_since_last_match(self, today: pd.Timestamp) -> int:
        """Return number of days since last mach."""
        return (today - self.date_last_match).days

    def update(self, match: Match, played_as: Team) -> None:
        """Update team data based on data from one match."""
        self.date_last_match = max(pd.to_datetime(match.Date), self.date_last_match)

        win = match.H if played_as == Team.Home else match.A
        points = match.HSC if played_as == Team.Home else match.ASC

        self.wins.put(win)
        self.scored.put(points)

        if played_as == Team.Home:
            self.wins_home.put(win)
            self.scored_home.put(points)
        else:
            self.wins_away.put(win)
            self.scored_away.put(points)

        # Update additional stats
        self.fgm.put(match.HFGM if played_as == Team.Home else match.AFGM)
        self.fga.put(match.HFGA if played_as == Team.Home else match.AFGA)
        self.fg3m.put(match.HFG3M if played_as == Team.Home else match.AFG3M)
        self.fg3a.put(match.HFG3A if played_as == Team.Home else match.AFG3A)
        self.ftm.put(match.HFTM if played_as == Team.Home else match.AFTM)
        self.fta.put(match.HFTA if played_as == Team.Home else match.AFTA)
        self.orb.put(match.HORB if played_as == Team.Home else match.AORB)
        self.drb.put(match.HDRB if played_as == Team.Home else match.ADRB)
        self.rb.put(match.HRB if played_as == Team.Home else match.ARB)
        self.ast.put(match.HAST if played_as == Team.Home else match.AAST)
        self.stl.put(match.HSTL if played_as == Team.Home else match.ASTL)
        self.blk.put(match.HBLK if played_as == Team.Home else match.ABLK)
        self.tov.put(match.HTOV if played_as == Team.Home else match.ATOV)
        self.pf.put(match.HPF if played_as == Team.Home else match.APF)

    def get_data_series(
        self, date: pd.Timestamp
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return complete data vector for given team."""
        vector_parameters = np.stack(
            [
                self.wins.values,
                self.wins_home.values,
                self.wins_away.values,
                self.scored.values,
                self.scored_home.values,
                self.scored_away.values,
                self.fgm.values,
                self.fga.values,
                self.fg3m.values,
                self.fg3a.values,
                self.ftm.values,
                self.fta.values,
                self.orb.values,
                self.drb.values,
                self.rb.values,
                self.ast.values,
                self.stl.values,
                self.blk.values,
                self.tov.values,
                self.pf.values,
            ],
        )
        # print(vector_parameters)
        scalar_parameters = np.array(
            [self._get_days_since_last_match(date)], dtype=np.float64
        )
        return vector_parameters, scalar_parameters


class Data:
    """Class for working with data."""

    def __init__(self) -> None:
        """Create Data."""
        self.teams: dict[TeamID, TeamData] = {}

    def add_match(self, match: Match) -> None:
        """Update team data based on data from one mach."""
        self.teams.setdefault(match.HID, TeamData()).update(match, Team.Home)
        self.teams.setdefault(match.AID, TeamData()).update(match, Team.Away)

    def team_data(self, team_id: TeamID) -> TeamData:
        """Return the TeamData for given team."""
        return self.teams[team_id]

    def get_match_parameters(
        self, match: Opp
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get array for match."""
        home_team = self.teams.setdefault(match.HID, TeamData())
        away_team = self.teams.setdefault(match.AID, TeamData())

        vector_home, scalar_home = home_team.get_data_series(match.Date)
        vector_away, scalar_away = away_team.get_data_series(match.Date)

        vector_parameters = np.concat(
            [
                vector_home,
                vector_away,
            ],
            dtype=np.float64,
        )

        scalar_parameters = np.concat(
            [scalar_home, scalar_away],
            dtype=np.float64,
        )

        return vector_parameters, scalar_parameters
