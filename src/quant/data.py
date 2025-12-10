from itertools import product, repeat, starmap
from operator import add
from itertools import chain


from quant.queue import RollingQueue
from quant.types import Match, Opp, Team, TeamID
import pandas as pd


class TeamData:
    """Hold data of one team, both as home and away."""

    N_SHORT = 5
    N_LONG = 30

    BASE_COLUMNS: tuple[str, ...] = (
        "WR",
        "WRH",
        "WRA",
        "PSA",
        "PSAH",
        "PSAA",
        "PLTA",
        "PLTAH",
        "PLTAA",
        "PD",
        "PDH",
        "PDA",
    )

    TEAM_COLUMNS: tuple[str, ...] = (
        "DSLM",
        *starmap(add, product(BASE_COLUMNS, ["_S", "_L"])),
    )

    # HACK: Python's scopes are weird, so we have to work around them with the
    # extra repeat iterator
    COLUMNS: tuple[tuple[str, ...], ...] = tuple(
        tuple(starmap(add, product(team_prefix, tc)))
        for team_prefix, tc in zip([["H_"], ["A_"]], repeat(TEAM_COLUMNS))
    )

    MATCH_COLUMNS: tuple[str, ...] = tuple(chain.from_iterable(COLUMNS))

    def __init__(
        self,
    ) -> None:
        """Init datastucture."""
        self.date_last_match: pd.Timestamp = pd.to_datetime("1977-11-10")

        # short averages
        self.win_rate_S = RollingQueue(TeamData.N_SHORT)
        self.win_rate_home_S = RollingQueue(TeamData.N_SHORT)
        self.win_rate_away_S = RollingQueue(TeamData.N_SHORT)

        self.points_scored_average_S = RollingQueue(TeamData.N_SHORT)
        self.points_scored_average_home_S = RollingQueue(TeamData.N_SHORT)
        self.points_scored_average_away_S = RollingQueue(TeamData.N_SHORT)

        self.points_lost_to_x_average_S = RollingQueue(TeamData.N_SHORT)
        self.points_lost_to_x_average_home_S = RollingQueue(TeamData.N_SHORT)
        self.points_lost_to_x_average_away_S = RollingQueue(TeamData.N_SHORT)

        self.points_diference_average_S = RollingQueue(TeamData.N_SHORT)
        self.points_diference_average_home_S = RollingQueue(TeamData.N_SHORT)
        self.points_diference_average_away_S = RollingQueue(TeamData.N_SHORT)

        # long averages
        self.win_rate_L = RollingQueue(TeamData.N_LONG)
        self.win_rate_home_L = RollingQueue(TeamData.N_LONG)
        self.win_rate_away_L = RollingQueue(TeamData.N_LONG)

        self.points_scored_average_L = RollingQueue(TeamData.N_LONG)
        self.points_scored_average_home_L = RollingQueue(TeamData.N_LONG)
        self.points_scored_average_away_L = RollingQueue(TeamData.N_LONG)

        self.points_lost_to_x_average_L = RollingQueue(TeamData.N_LONG)
        self.points_lost_to_x_average_home_L = RollingQueue(TeamData.N_LONG)
        self.points_lost_to_x_average_away_L = RollingQueue(TeamData.N_LONG)

        self.points_diference_average_L = RollingQueue(TeamData.N_LONG)
        self.points_diference_average_home_L = RollingQueue(TeamData.N_LONG)
        self.points_diference_average_away_L = RollingQueue(TeamData.N_LONG)

    def _get_days_since_last_mach(self, today: pd.Timestamp) -> int:
        """Return number of days scince last mach."""
        return (today - self.date_last_match).days

    def update(self, match: Match, played_as: Team) -> None:
        """Update team data based on data from one mach."""
        self.date_last_match = max(pd.to_datetime(match.Date), self.date_last_match)

        win = match.H if played_as == Team.Home else match.A
        points = match.HSC if played_as == Team.Home else match.ASC
        points_lost_to = match.ASC if played_as == Team.Home else match.HSC
        point_diference = points - points_lost_to

        self.win_rate_S.put(win)
        self.win_rate_L.put(win)
        self.points_scored_average_S.put(points)
        self.points_scored_average_L.put(points)
        self.points_lost_to_x_average_S.put(points_lost_to)
        self.points_lost_to_x_average_L.put(points_lost_to)
        self.points_diference_average_S.put(point_diference)
        self.points_diference_average_L.put(point_diference)

        if played_as == Team.Home:
            self.win_rate_home_S.put(win)
            self.win_rate_home_L.put(win)
            self.points_scored_average_home_S.put(points)
            self.points_scored_average_home_L.put(points)
            self.points_lost_to_x_average_home_S.put(points_lost_to)
            self.points_lost_to_x_average_home_L.put(points_lost_to)
            self.points_diference_average_home_S.put(point_diference)
            self.points_diference_average_home_L.put(point_diference)
        else:
            self.win_rate_away_S.put(win)
            self.win_rate_away_L.put(win)
            self.points_scored_average_away_S.put(points)
            self.points_scored_average_away_L.put(points)
            self.points_lost_to_x_average_away_S.put(points_lost_to)
            self.points_lost_to_x_average_away_L.put(points_lost_to)
            self.points_diference_average_away_S.put(point_diference)
            self.points_diference_average_away_L.put(point_diference)

    def get_data_series(self, date: pd.Timestamp, team: Team) -> pd.Series:
        """Return complete data vector for given team."""
        return pd.Series(
            [
                self._get_days_since_last_mach(date),
                self.win_rate_S.average(),
                self.win_rate_L.average(),
                self.win_rate_home_S.average(),
                self.win_rate_home_L.average(),
                self.win_rate_away_S.average(),
                self.win_rate_away_L.average(),
                self.points_scored_average_S.average(),
                self.points_scored_average_L.average(),
                self.points_scored_average_home_S.average(),
                self.points_scored_average_away_L.average(),
                self.points_scored_average_home_L.average(),
                self.points_scored_average_away_S.average(),
                self.points_lost_to_x_average_S.average(),
                self.points_lost_to_x_average_L.average(),
                self.points_lost_to_x_average_home_S.average(),
                self.points_lost_to_x_average_home_L.average(),
                self.points_lost_to_x_average_away_S.average(),
                self.points_lost_to_x_average_away_L.average(),
                self.points_diference_average_S.average(),
                self.points_diference_average_L.average(),
                self.points_diference_average_home_S.average(),
                self.points_diference_average_home_L.average(),
                self.points_diference_average_away_S.average(),
                self.points_diference_average_away_L.average(),
            ],
            index=pd.Index(self.COLUMNS[team]),
        )


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

    def get_match_parameters(self, match: Opp) -> pd.Series:
        """Get array for match."""
        home_team = self.teams.setdefault(match.HID, TeamData())
        away_team = self.teams.setdefault(match.AID, TeamData())

        date: pd.Timestamp = pd.to_datetime(match.Date)

        return pd.concat(
            [
                home_team.get_data_series(date, Team.Home),
                away_team.get_data_series(date, Team.Away),
            ]
        )
