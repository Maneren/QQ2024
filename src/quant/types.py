from collections import namedtuple
from enum import IntEnum


type TeamID = int

Match = namedtuple(
    "Match",
    [
        "Index",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "OddsH",
        "OddsA",
        "H",
        "A",
        "HSC",
        "ASC",
        "HFGM",
        "AFGM",
        "HFGA",
        "AFGA",
        "HFG3M",
        "AFG3M",
        "HFG3A",
        "AFG3A",
        "HFTM",
        "AFTM",
        "HFTA",
        "AFTA",
        "HORB",
        "AORB",
        "HDRB",
        "ADRB",
        "HRB",
        "ARB",
        "HAST",
        "AAST",
        "HSTL",
        "ASTL",
        "HBLK",
        "ABLK",
        "HTOV",
        "ATOV",
        "HPF",
        "APF",
    ],
    defaults=(None,) * 32,
)

Opp = namedtuple(
    "Opp",
    [
        "Index",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "OddsH",
        "OddsA",
        "BetH",
        "BetA",
    ],
)


def match_to_opp(match: Match) -> Opp:
    """
    Convert Match to Opp.

    Fills Bets with 0.
    """
    return Opp(
        Index=match.Index,
        Season=match.Season,
        Date=match.Date,
        HID=match.HID,
        AID=match.AID,
        N=match.N,
        POFF=match.POFF,
        OddsH=match.OddsH,
        OddsA=match.OddsA,
        BetH=0,
        BetA=0,
    )


Summary = namedtuple(
    "Summary",
    [
        "Bankroll",
        "Date",
        "Min_bet",
        "Max_bet",
    ],
)


class Team(IntEnum):
    """Enum discerning teams playing home or away."""

    Home = 0
    Away = 1
