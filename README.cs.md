# Qminers Quant Hackathon 2024

## Struktura repozitáře

- `problem_info.md` obsahuje detailní [zadání](problem_info.md) úlohy včetně popisu dat
- `data/games.csv` a `data/players.csv` obsahují trénovací data.
- `src/main.py` skript pro spuštění lokáního tréninku a vyhodnocení
- `src/quant` obsahuje implementaci modelu
  - `quant/model.py` hlavní kód modelu
    - `Model` - organizuje zpracování dat a trénovaní modelu
  - `quant/data.py` třídy pro práci s načtenými daty
  - zbytek jsou různé pomocné třídy a funkce

## Spuštění

Vyžaduje alespoň Python 3.13 a správce balíčků s podporou PEP508, např. [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync
uv run src/main.py
```

> [!note]]
> První spuštění zpracuje vstupní data a uloží je do souboru
> `data/training_data.npz`, další spuštení z něj data pouze čtou a znovu
> nezpracovávají.
