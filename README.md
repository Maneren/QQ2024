# Qminers Quant Hackathon 2024

## Repository Structure

- `problem_info.md` contains detailed [problem statement](problem_info.md) including data description
- `data/games.csv` and `data/players.csv` contain training data.
- `src/main.py` script for running local training and evaluation
- `src/quant` contains model implementation
  - `quant/model.py` main model code
    - `Model` - organizes data processing and model training
  - `quant/data.py` classes for working with loaded data
  - the rest are various helper classes and functions

## Running

Requires at least Python 3.13 and a package manager with PEP508 support, e.g.
[`uv`](https://docs.astral.sh/uv/). Then run as follows:

```sh
uv sync
uv run src/main.py
```

> [!note]
> First run will process input data and save it to the file
> `data/training_data.npz`, subsequent runs will only read data from it and
> won't reprocess it.
