# Strathmore Lost & Found AI

This is the **machine-learning part** of a proposed Strathmore University Lost & Found mobile application. Given one lost-item report and one found-item report, it estimates whether staff should review them as a possible match. It does **not** establish ownership or automatically return an item.

- **AI Application:** Classification
- **Algorithm/Approach:** Logistic Regression
- **Current output:** a match probability and a binary `match` / `non-match` suggestion for each report pair. Several found reports can then be ordered by probability for staff review.

The [main learning notebook](notebooks/LogisticRegression_Base.ipynb) now shows real Delhi found-item exploration followed by a **constructed** lost-claim matching exercise. Small helper modules keep cleaning and feature calculations consistent between the notebook and command-line run.

## Current Delhi experiment and data lineage

```text
Real Delhi found records (13,713; local raw CSV unchanged)
  → cleaned found rows (13,713; generated, local)
  → 120 fictional lost claims derived from selected found rows
  → 360 constructed pairs (120 positives, 240 assumed negatives)
  → numeric pair features → Logistic Regression → grouped validation
```

The found-item records are real. The lost claims and pair labels are **synthetic academic data**, not reports from Delhi passengers or Strathmore students. Different source rows are not necessarily different physical belongings, so the negatives are assumptions. This experiment cannot measure real lost-and-found matching accuracy. The Delhi raw CSV and all row-level derivatives are local and ignored by Git because descriptions may contain personal information. Station cleaning is limited to whitespace and casefolded helper columns; no unverified station aliases are merged. See [dataset sources](docs/dataset_sources.md) for the source and expected hash.

The earlier synthetic campus experiment remains in the repository as a small comparative teaching baseline. Its saved metrics, described below, are separate from the new Delhi exercise.

## Retained older campus dataset

We use two tracked **synthetic** campus files:

| File | What it contains | Used for |
|---|---|---|
| [`synthetic_campus_reports.csv`](data/raw/campus/synthetic_campus_reports.csv) | 46 lost/found report examples | Exploration, cleaning, and pair features |
| [`labelled_campus_pairs.csv`](data/raw/campus/labelled_campus_pairs.csv) | 32 labelled lost/found pairs: 14 `match`, 10 `non_match`, 8 `plausible` | Training and evaluation |

The older raw campus reports contain six shifted rows caused by an extra empty CSV field. Cleaning writes a corrected **copy** under `data/processed/`; the raw files stay unchanged. `plausible` means an unconfirmed possible match. That older experiment maps it to class 0, but also checks results with those uncertain rows excluded.

The Delhi Metro data is now used for real-data cleaning and EDA. Synthetic claims made from selected Delhi rows provide the main notebook's constructed model exercise. No image dataset or image model is included. The table above describes the retained, older campus inputs, which can still be regenerated with `python -m scripts.run_model`.

## Repository map

```text
SemProject_MLEngine/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/campus/              # retained synthetic campus CSV files
│   ├── raw/delhi/               # local, ignored real source CSV
│   └── processed/               # cleaned rows, claims, pair features, split
├── notebooks/
│   └── LogisticRegression_Base.ipynb
├── matching_system/             # loading, text cleaning, features, model
├── scripts/                     # short commands for each data/model step
├── results/
│   ├── figures/                 # aggregate Delhi and older campus charts
│   ├── delhi_experiment/        # local, ignored row-level model output
│   └── ml/                      # older campus predictions and metrics
├── tests/                       # focused checks of important behavior
└── docs/dataset_sources.md
```

The model code is in [`matching_system/ml_model.py`](matching_system/ml_model.py). [`feature_builder.py`](matching_system/feature_builder.py) turns a pair of reports into numeric inputs. Start with the notebook before reading the helper code. `scripts/prepare_delhi_data.py` and `scripts/create_delhi_experiment.py` reproduce the Delhi path.

## Install and run

Use Python 3.11 or newer. From the repository root:

```powershell
git clone https://github.com/nimrod-kobia/SemProject_MLEngine.git
cd SemProject_MLEngine
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook
```

Open `notebooks/LogisticRegression_Base.ipynb` in Jupyter and choose **Run All** after adding the local Delhi CSV as described next.
In VS Code, open this repository folder, then use the notebook's **Select Kernel** menu to choose the Python environment in `.venv`. The workspace setting in `.vscode/settings.json` suggests that environment for new workspaces, but VS Code can retain a kernel selected earlier. If an import fails, run `import sys; print(sys.executable)` in a cell to check which Python is actually running.
For the **current Delhi notebook**, first obtain the source CSV linked in [dataset sources](docs/dataset_sources.md), place it at `data/raw/delhi/delhi_lost_and_found.csv`, and check the documented SHA-256. Then run `python -m scripts.create_delhi_experiment` and choose **Run All**. The notebook shows real missingness, duplicates, date and whitespace checks, four aggregate EDA figures, synthetic-claim generation, pair features, model fitting, grouped validation and limitations. Row-level generated Delhi files remain ignored by Git.

To check the generated pair labels and split without downloading the Delhi data, run `python -m unittest tests.test_delhi_validator`. To validate local generated files after creating them, run `python -m scripts.validate_delhi_experiment`. This checks three candidates per synthetic claim (one constructed positive and two assumed negatives), unique found IDs, split separation and coverage, and pair counts in the split manifest. These checks catch structural generation and leakage mistakes; they cannot establish that distinct Delhi source rows represent different physical belongings or verify assumed labels.

To regenerate the **older campus baseline** without Jupyter:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_model
```

On macOS/Linux, activate `.venv` with `source .venv/bin/activate` and use `python` in place of the Windows executable path.

## How the model works

```text
Real Delhi found rows + synthetic lost claims + constructed pair labels
    → preserve source rows, clean whitespace and parse dates
    → compare each lost/found pair using 14 numeric features
    → split by lost-report ID
    → fit StandardScaler + Logistic Regression on development pairs
    → grouped cross-validation and held-out test
    → probabilities, predictions, and error analysis
```

The features compare description words, locations, item attributes, and reported times. A missing attribute produces neither a match nor a conflict. In the current notebook, `y` is 1 for a constructed positive and 0 for an assumed negative; `X` is the feature table. Location contributes to the score without excluding a candidate. The older campus exercise also has `plausible` labels, handled in its separate results.

All candidate pairs belonging to one lost report remain in the same split. This keeps the same lost description from appearing in both training and testing. Scaling is learned inside each training fold before that fold's validation pairs are checked.

## Measured results and example

The current Delhi-derived held-out check has **108 constructed pairs**. Its precision is **0.971**, recall **0.944**, and F1 **0.958**, with confusion matrix `[[71, 1], [2, 34]]`. These numbers only measure the generated-pair exercise. The older campus exercise's eight-pair test gave precision 1.00, recall 0.75 and F1 0.857; its saved file is [`validation_metrics.json`](results/ml/validation_metrics.json). Neither result estimates real-campus accuracy.

The notebook displays sample predictions by report ID so that potentially sensitive source descriptions do not enter public outputs. Staff must inspect any suggested pair and verify ownership privately.

## Limits and next work

- Delhi lost claims and labels are constructed; no real handover labels exist. The separate older campus example has only 32 designed pairs.
- Mapping `plausible` to 0 in that older campus example is an assumption; its model run includes a check excluding those rows.
- There is no real campus field test and no validated photo matching. The trained classifier is offline; the Android app does not use it yet.
- Next: review labels, gather consented campus examples, compare another simple model using the same grouped split and metrics, and integrate tested suggestions into the app with staff verification.

## CAT 2 Assessment Guide

CAT 2 is the **assessment**, not the name of the model or a software component. The brief asks for the following evidence:

| Assessment requirement | Evidence in this repository | Status |
|---|---|---|
| **1. Identify the ML approach and algorithms relevant to the dataset and expected app output; enter them in the shared spreadsheet.** | This README and notebook sections **1 and 6** explain pair classification, Logistic Regression, inputs and review output. | Repository explanation complete. The spreadsheet entry must be checked separately. |
| **2. Critically detail cleaning from exploration to the final modelling dataset; derive at least four EDA observations.** | Notebook sections **2–6** show raw Delhi fields, missingness, duplicate/date/whitespace checks, cleaned audit, four real-data observations and figures, then constructed pair features. | Complete for the available Delhi copy, with the synthetic-label limit stated. |
| **3. Describe the base model and apply suitable validity measures, such as cross-validation for classification.** | Notebook sections **6–8** show `X`, `y`, grouped cross-validation, held-out metrics, confusion matrix and sample predictions. | Complete for constructed labels; real-case accuracy unmeasured. |
| **4. Determine further methods to validate the base model.** | Notebook section **9** describes independently verified handovers, real-item holdouts, a fair model comparison, top-three retrieval and error review. The brief says a description is sufficient now. | Described; real-case comparison is future work. |

The required notebook filename is `LogisticRegression_Base.ipynb`, and the public submission repository is `SemProject_MLEngine`. The group must still check its spreadsheet entry, members' forks, assigned presentation date, and each member's ability to explain the work. Follow the course's rules on permitted AI assistance and authorship.

Earlier work by team members is in the companion `AI-project` history. See [team work and Git history](docs/contributions.md) for verifiable commits and how fork pull requests will make future work visible in this submission repository.

### Presentation path

1. Open this README and state the problem, dataset, model, and expected app output.
2. Open the main notebook. Show the original Delhi columns, missing values and cleaning audit.
3. Explain the four real-data observations and charts.
4. Show one row of pair features, identify `X` and `y`, and point to `model.fit`.
5. Explain why the split and validation are grouped by lost report.
6. Show the held-out confusion matrix and example predictions, making clear the labels were constructed.
7. State the limitations and how the team would validate with verified real handovers before app integration.
