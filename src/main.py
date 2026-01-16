from itertools import starmap
import pandas as pd
import numpy as np
from quant.data import Data, TeamData
from quant.files import (
    DATASET_NAME,
    DatasetTensors,
    load_cached_dataset,
    load_source_data,
    save_dataset,
)
from quant.model import MATCH_SCALAR_FEATURES, MATCH_VECTOR_FEATURES, Model
from quant.ranking import Elo, EloByLocation
from quant.types import Match, Opp, match_to_opp
from tqdm import tqdm


def dataframe_to_tensors(dataframe: pd.DataFrame) -> DatasetTensors:
    """Convert dataframe to tensors."""
    np.set_printoptions(edgeitems=30, linewidth=240)
    data = Data()
    elo = Elo()
    elo_by_location = EloByLocation()

    def get_match_parameters(
        match: Opp,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get parameters for given match."""
        match_elo = elo.match_rating(match.HID, match.AID)
        match_elo_by_location = elo_by_location.match_rating(match.HID, match.AID)

        vector_parameters, scalar_parameters = data.get_match_parameters(match)

        scalar_parameters = np.concatenate(
            [
                [*match_elo, *match_elo_by_location],
                scalar_parameters,
            ],
            axis=None,
            dtype=np.float64,
        )

        return vector_parameters, scalar_parameters

    samples = len(dataframe)
    nd_data = np.zeros((samples, len(MATCH_VECTOR_FEATURES), TeamData.N))
    scalar_data = np.zeros((samples, len(MATCH_SCALAR_FEATURES)))
    outcomes = np.zeros((samples,))

    for i, match in tqdm(
        enumerate(starmap(Match, dataframe.itertuples())),
        total=samples,
        desc="Creating training dataframe",
    ):
        nd, scalar = get_match_parameters(match_to_opp(match))

        nd_data[i] = nd
        scalar_data[i] = scalar
        outcomes[i] = np.float64(match.HSC - match.ASC)

        data.add_match(match)
        elo.add_match(match)
        elo_by_location.add_match(match)

    return nd_data, scalar_data, outcomes


def get_dataset() -> DatasetTensors:
    """Train AI."""

    if (dataset := load_cached_dataset(DATASET_NAME)) is not None:
        print("Loading cached dataset...")
        return dataset

    print("Loading source data...")
    dataframe = load_source_data()

    dataset = dataframe_to_tensors(dataframe)
    print("Saving dataset...")
    save_dataset(DATASET_NAME, *dataset)

    return dataset


def main():
    dataset = get_dataset()

    model = Model()
    model.train_reg(*dataset)


if __name__ == "__main__":
    main()
