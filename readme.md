# Policies of Multiple Skill Levels for Better Strength Estimation in Games

This repository contains the official implementation of the paper:

**"Policies of Multiple Skill Levels for Better Strength Estimation in Games"**  
*AIIDE 2025*

**Authors:** Kyota Kuboki, Tatsuyoshi Ogawa, Chu-Hsuan Hsueh, Shi-Jim Yen, Kokolo Ikeda  
Paper: *(link to camera-ready; add when available)*

---

## Overview

We propose a method to estimate player strength from a **small number of matches** in **Go** and **Chess**.  
Our method integrates three sources:

1. **Strength scores (s-scores)** from Chen et al. (2025)  
2. **Policies** (priors) from imitation models across different skill levels (e.g., KataGo HumanSL / Maia)  
3. **Advantage losses** computed from strong AI evaluations (KataGo / LeelaChessZero)

By training a LightGBM meta-model on these features, we achieve state-of-the-art performance:
- **Go:** 91.7% accuracy with 20 matches  
- **Chess:** 88.2% accuracy with 20 matches

---

## Repository Structure

```
.
├── chess/ # Chess pipeline
│ ├── select_game_record.py # Select game records into splits
│ ├── pgn_analyzer.py # Analyze with Lc0 (value/prior) + Maia (prior)
│ ├── make_turn_info.py # Build turn-level features
│ ├── make_game_info.py # Aggregate to game-level features
│ ├── training/ # Data split: training
│ ├── candidating/ # Data split: for Chen et al.’s estimator
│ ├── testing/ # Data split: rank-group evaluation
│ └── testing_by_player/ # Data split: player-specific evaluation
│
├── go/ # Go pipeline
│ ├── select_game_record.py
│ ├── sgf_analyzer.py # Analyze with KataGo (value/prior) + HumanSL (prior)
│ ├── make_turn_info.py
│ ├── make_game_info.py
│ ├── analysis_deterministic_analyzer.cfg # KataGo config (deterministic analyzer)
│ ├── training/
│ ├── candidating/
│ ├── testing/
│ └── testing_by_player/
│
├── evaluator_chen_method.py # Evaluate Chen et al.’s baseline
├── evaluator_my_method.py # Train/Eval our proposed method
├── requirements.txt
└── README.md
```

---

## Requirements

Install via pip:

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.10+
- LightGBM, NumPy, Pandas
- (External engines/models are **not** bundled; see below)

### Lco Python bindings (required for `chess/pgn_analyzer.py`)

`pgn_analyzer.py` uses **Leela Chess Zero** from Python. You need the Lc0 Python bindings:

#### Option A (recommended): install from Git (builds `lczero-bindings`)
```bash
pip install --user git+https://github.com/LeelaChessZero/lc0.git
```
This builds and installs the `lczero-bindings` package so that `import lczero.backends` works.

#### Option B (PyPl, may lag behind Git)
```bash
pip install lczero-bindings
```

> We originally followed a Python binding approach documented in [LCZero pull request](https://github.com/LeelaChessZero/lc0/pull/1261). The official repository now exposes a straightforward `pip install git+...` path. After installing the bindings, remember to place the **Lc0 weights and Maia model files** in the Lc0 `scripts/` directory as described in the *Analyze Records* section.

---

## Data

We use public datasets:
- **Go:** [Fox Go dataset](https://github.com/featurecat/go-dataset) — SGF records (download & extract under `go/`)
- **Chess:** [Lichess database](https://database.lichess.org/) — Standard, rated PGN for **February 2024** (blitz)

Selection criteria follow the paper (e.g., minimum plies, valid terminations, both players in the same rank/rating group).  
Place raw downloads under the respective game directory before Step 1.

---

## External Models

This project **depends** on external AI engines/models for analysis and priors. Please obtain them from their official sources.

- **[KataGo](https://github.com/lightvector/KataGo)** (engine + strong AI weights; we used `kata1-b18c384nbt-s9732312320-d4245566942.bin.gz`)
- **[KataGo HumanSL](https://github.com/lightvector/KataGo/releases/tag/v1.15.0)** (imitation model; we used `b18c384nbt-humanv0.bin.gz`)
- **[LeelaChessZero (Lc0)](https://github.com/LeelaChessZero/lc0)** (engine + weights; we used  
  `195b450999e874d07aea2c09fd0db5eff9d4441ec1ad5a60a140fe8ea94c4f3a`)
- **[Maia](https://github.com/CSSLab/maia-chess)** (imitation models; e.g., `maia-1100.pt`, …, `maia-1900.pt`)
- **[Chen et al.’s strength estimator](https://github.com/rlglab/strength-estimator/)** (code to train s-scores)

> We do **not** redistribute these engines or weights. Follow each repository’s installation instructions and licenses.

---

## Usage

### 1) Select Game Records

This step filters raw records into four splits — `training/`, `candidating/`, `testing/`, and `testing_by_player/`.

**Chess (Lichess)**
1. Download the *standard, rated* PGN for **Feb 2024** and extract (file should look like `lichess_db_standard_rated_2024-02.pgn`).
2. Run:
   ```bash
   # from repo root
   python chess/select_game_record.py -t chess/lichess_db_standard_rated_2024-02.pgn
   # or from within chess/
   cd chess
   python select_game_record.py -t lichess_db_standard_rated_2024-02.pgn
   ```
3. Outputs under `chess/`:
   ```
   training/  candidating/  testing/  testing_by_player/
   ```

**Go (Fox Go)**
1. Download & extract Fox Go under `go/` (e.g., `go/FoxGo/` contains SGF files grouped by dan-kyu).
2. Run:
   ```bash
   # from repo root
   python go/select_game_record.py -t go/FoxGo
   # or from within go/
   cd go
   python select_game_record.py -t FoxGo
   ```
3. Outputs under `go/`:
   ```
   training/  candidating/  testing/  testing_by_player/
   ```

> Notes  
> - Ensure PGN/SGF inputs are decompressed (multi-GB possible).  
> - Splits and grouping follow the paper’s criteria.

---

### 2) Analyze Records

We perform three analyses per game.

**Chess**
1) Train **Chen et al.** estimator → export **s-scores** (CSV)  
2) Use **Lc0** → **value/prior**  
3) Use **Maia** → **prior**  
→ (2) and (3) are done by `chess/pgn_analyzer.py`.

**Go**
1) Train **Chen et al.** estimator → export **s-scores** (CSV)  
2) Use **KataGo** → **value/prior**  
3) Use **KataGo HumanSL** → **prior**  
→ (2) and (3) are done by `go/sgf_analyzer.py`.

**Chess: Lc0 & Maia (`pgn_analyzer.py`)**
- Edit **`chess/pgn_analyzer.py` lines 9–22**:
  - **Line 9:** absolute path to **Lc0 `scripts/` directory**
  - **Lines 11–22:** filenames for **Maia models** and **Lc0 weights**
  - Place **Maia models** and **Lc0 weights** **inside** the Lc0 `scripts/` directory
  - We used Lc0 weights:  
    `195b450999e874d07aea2c09fd0db5eff9d4441ec1ad5a60a140fe8ea94c4f3a`
- Run (example for training group):
  ```bash
  # from chess/
  python pgn_analyzer.py -t training/1000-1200.pgn
  # or from repo root
  python chess/pgn_analyzer.py -t chess/training/1000-1200.pgn
  ```
- Output (for `training/1000-1200.pgn`):
  ```
  chess/training_analyzed/
    <game_id>.json.gz  # per-game analysis (Lc0 value/prior + Maia prior)
  ```

**Go: KataGo & HumanSL (`sgf_analyzer.py`)**
- Edit **`go/go_engines.py` lines 12–13**:
  ```python
  KATAGO_BIN = "/abs/path/to/katago"      # KataGo executable
  KATAGO_DIR = "/abs/path/to/katago_dir"  # contains configs/ and weights
  ```
  - Weights used:
    - KataGo (strong AI): `kata1-b18c384nbt-s9732312320-d4245566942.bin.gz`
    - HumanSL: `b18c384nbt-humanv0.bin.gz`
- Run (example for training group):
  ```bash
  # from go/
  python sgf_analyzer.py -t training/3-5k.sgf -n 2 --no-move-owner --skip-analyzed-sgf     -kth preaz_9d+preaz_7d+preaz_5d+preaz_3d+preaz_1d+preaz_2k+preaz_4k+preaz_6k+preaz_8k+preaz_10k
  # or from repo root
  python go/sgf_analyzer.py -t go/training/3-5k.sgf -n 2 --no-move-owner --skip-analyzed-sgf     -kth preaz_9d+preaz_7d+preaz_5d+preaz_3d+preaz_1d+preaz_2k+preaz_4k+preaz_6k+preaz_8k+preaz_10k
  ```
- Output (for `training/3-5k.sgf`):
  ```
  go/training/3-5k_analyzed/
    <game_id>.json.gz  # per-game analysis (KataGo value/prior + HumanSL prior)
  ```

**Chen et al.’s s-scores**
- Follow Chen et al.’s repo to train on the `training/` split and export s-scores per group.
- Expected outputs:
  - Chess: `chess/training/1000-1200.csv`, …
  - Go: `go/training/3-5k.csv`, `go/training/1-2k.csv`, `go/training/1d.csv`, …

---

### 3) Extract Features

From the three sources (s-scores, strong AI value/prior, imitation priors), we build **turn-level** then **game-level** features.

**Turn-level features**
```bash
# choose one split: -t {training|candidating|testing|testing_by_player}

# Chess
cd chess
python make_turn_info.py -t training

# Go
cd ../go
python make_turn_info.py -t training
```
- Inputs:
  - Chess: `chess/training_analyzed/*.json.gz`, `chess/training/*.csv` (s-scores)
  - Go: `go/training/*_analyzed/*.json.gz`, `go/training/*.csv` (s-scores)
- Outputs (examples):
  - Chess: `chess/training/1000_1200_turn_info.csv`
  - Go:   `go/training/3_5k_turn_info.csv`

**Game-level features**
```bash
# Chess
cd chess
python make_game_info.py -t training

# Go
cd ../go
python make_game_info.py -t training
```
- Inputs: the corresponding `*_turn_info.csv`
- Outputs (examples):
  - Chess: `chess/training/1000_1200_game_info.csv`
  - Go:   `go/training/3_5k_game_info.csv`

---

### 4) Train & Evaluate

Two evaluators:

- `evaluator_my_method.py` — **Proposed method**
- `evaluator_chen_method.py` — **Chen et al. baseline**

Both accept `-t {chess|go}`.

**Proposed method**
```bash
python evaluator_my_method.py -t chess
python evaluator_my_method.py -t go
```

**Chen et al. baseline**
```bash
python evaluator_chen_method.py -t chess
python evaluator_chen_method.py -t go
```

**Behavior**
- Trains LightGBM on `training/` and evaluates:
  - **Rank-group sampling**: `testing/`
  - **Player-specific sampling**: `testing_by_player/`
- Reports **Accuracy** and **Accuracy±1** for multiple `n` (e.g., 1, 5, 10, 15, 20).

**Outputs**
```
results/
  chess/
    chen_method_1.csv
    chen_method_5.csv
    ...
    accuracy_info_chen_method.csv
    all_columns.csv
    without_s-score.csv
    ...
    confusion_matrix_all_columns_1.png
    confusion_matrix_all_columns_5.png
    ...
  go/
    ...
```

---

## Results

Main results reproduced from the paper (Table 2):

- **Go:** 91.7% accuracy with 20 matches  
- **Chess:** 88.2% accuracy with 20 matches

### Reproducibility Note

In the paper, **LightGBM’s random seed was not fixed**. Therefore, exact numbers may **not** match when you re-run experiments, although results should be **similar in magnitude**.

*(This repository fixed seed for LightGBM.)*

---