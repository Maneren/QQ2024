from pathlib import Path
import numpy as np
import pandas as pd

DatasetTensors = tuple[np.ndarray, np.ndarray, np.ndarray]

DATA_FOLDER = Path("data")
DATASET_NAME = "training_data"


def dataset_path(name: str) -> Path:
    return DATA_FOLDER / f"{name}.npz"


def load_cached_dataset(name: str) -> DatasetTensors | None:
    """Load cached dataset."""
    file = dataset_path(name)
    if not file.is_file():
        return None
    with np.load(file) as loaded:
        return loaded["nd_data"], loaded["scalar_data"], loaded["outcomes"]


def save_dataset(
    name: str, nd_data: np.ndarray, scalar_data: np.ndarray, outcomes: np.ndarray
) -> None:
    """Save dataset."""
    file = dataset_path(name)
    np.savez_compressed(
        file, nd_data=nd_data, scalar_data=scalar_data, outcomes=outcomes
    )


def load_source_data() -> pd.DataFrame:
    """Load source data."""
    dataframe = pd.read_csv(DATA_FOLDER / "games.csv", index_col=0)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe.drop(columns=["Open"], inplace=True)

    return dataframe
