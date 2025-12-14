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
    return tuple(starmap(add, product(prefixes, columns)))


class TeamData:
    """Hold data of one team, both as home and away."""

    N = 30

    TEAM_VECTOR_COLUMNS: tuple[str, ...] = (
        "WR",
        "WRH",
        "WRA",
        "PSA",
        "PSAH",
        "PSAA",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
        "ORB",
        "DRB",
        "RB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
    )

    TEAM_SCALAR_COLUMNS: tuple[str, ...] = ("DSLM",)

    MATCH_VECTOR_COLUMNS: tuple[str, ...] = prefix_columns(
        ["H_", "A_"], TEAM_VECTOR_COLUMNS
    )

    MATCH_SCALAR_COLUMNS = prefix_columns(["H_", "A_"], TEAM_SCALAR_COLUMNS)

    def __init__(
        self,
    ) -> None:
        """Init datastucture."""
        self.date_last_match: pd.Timestamp = pd.to_datetime("1975-11-07")

        self.win_rate = RollingQueue(TeamData.N)
        self.win_rate_home = RollingQueue(TeamData.N)
        self.win_rate_away = RollingQueue(TeamData.N)

        self.points_scored_average = RollingQueue(TeamData.N)
        self.points_scored_average_home = RollingQueue(TeamData.N)
        self.points_scored_average_away = RollingQueue(TeamData.N)

        # Field goals made/attempted
        self.fgm_average = RollingQueue(TeamData.N)
        self.fga_average = RollingQueue(TeamData.N)

        # 3-point field goals made/attempted
        self.fg3m_average = RollingQueue(TeamData.N)
        self.fg3a_average = RollingQueue(TeamData.N)

        # Free throws made/attempted
        self.ftm_average = RollingQueue(TeamData.N)
        self.fta_average = RollingQueue(TeamData.N)

        # Rebounds
        self.orb_average = RollingQueue(TeamData.N)
        self.drb_average = RollingQueue(TeamData.N)
        self.rb_average = RollingQueue(TeamData.N)

        # Other stats
        self.ast_average = RollingQueue(TeamData.N)
        self.stl_average = RollingQueue(TeamData.N)
        self.blk_average = RollingQueue(TeamData.N)
        self.tov_average = RollingQueue(TeamData.N)
        self.pf_average = RollingQueue(TeamData.N)

    def _get_days_since_last_match(self, today: pd.Timestamp) -> int:
        """Return number of days scince last mach."""
        return (today - self.date_last_match).days

    def update(self, match: Match, played_as: Team) -> None:
        """Update team data based on data from one mach."""
        self.date_last_match = max(pd.to_datetime(match.Date), self.date_last_match)

        win = match.H if played_as == Team.Home else match.A
        points = match.HSC if played_as == Team.Home else match.ASC

        self.win_rate.put(win)
        self.points_scored_average.put(points)

        if played_as == Team.Home:
            self.win_rate_home.put(win)
            self.points_scored_average_home.put(points)
        else:
            self.win_rate_away.put(win)
            self.points_scored_average_away.put(points)

        # Update additional stats
        self.fgm_average.put(match.HFGM if played_as == Team.Home else match.AFGM)
        self.fga_average.put(match.HFGA if played_as == Team.Home else match.AFGA)
        self.fg3m_average.put(match.HFG3M if played_as == Team.Home else match.AFG3M)
        self.fg3a_average.put(match.HFG3A if played_as == Team.Home else match.AFG3A)
        self.ftm_average.put(match.HFTM if played_as == Team.Home else match.AFTM)
        self.fta_average.put(match.HFTA if played_as == Team.Home else match.AFTA)
        self.orb_average.put(match.HORB if played_as == Team.Home else match.AORB)
        self.drb_average.put(match.HDRB if played_as == Team.Home else match.ADRB)
        self.rb_average.put(match.HRB if played_as == Team.Home else match.ARB)
        self.ast_average.put(match.HAST if played_as == Team.Home else match.AAST)
        self.stl_average.put(match.HSTL if played_as == Team.Home else match.ASTL)
        self.blk_average.put(match.HBLK if played_as == Team.Home else match.ABLK)
        self.tov_average.put(match.HTOV if played_as == Team.Home else match.ATOV)
        self.pf_average.put(match.HPF if played_as == Team.Home else match.APF)

    def get_data_series(
        self, date: pd.Timestamp
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return complete data vector for given team."""
        vector_parameters = np.stack(
            [
                self.win_rate.values,
                self.win_rate_home.values,
                self.win_rate_away.values,
                self.points_scored_average.values,
                self.points_scored_average_home.values,
                self.points_scored_average_away.values,
                self.fgm_average.values,
                self.fga_average.values,
                self.fg3m_average.values,
                self.fg3a_average.values,
                self.ftm_average.values,
                self.fta_average.values,
                self.orb_average.values,
                self.drb_average.values,
                self.rb_average.values,
                self.ast_average.values,
                self.stl_average.values,
                self.blk_average.values,
                self.tov_average.values,
                self.pf_average.values,
            ],
        )
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

        date = pd.to_datetime(match.Date)

        vector_home, scalar_home = home_team.get_data_series(date)
        vector_away, scalar_away = away_team.get_data_series(date)

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
