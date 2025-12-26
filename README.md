# Qminers Quant Hackathon 2024

## Struktura repozitáře

- `problem_info.md` obsahuje detailní [zadání](problem_info.md) úlohy včetně popisu dat
- `data/games.csv` a `data/players.csv` obsahují trénovací data.
- `src/environment.py` obsahuje evaluační smyčku
- `src/evaluate.py` skript pro spuštění lokáního vyhodnocení tvého modelu na trénovacích datech.
- `src/quant` obsahuje implementaci modelu
  - `quant/model.py` hlavní kód modelu
    - `Model` - organizuje zpracování dat a trénovaní modelu
    - `AI` - obal okolo tensorflow modelu
  - `quant/data.py` třídy pro práci s načtenými daty
  - zbytek jsou různé pomocné třídy a funkce
