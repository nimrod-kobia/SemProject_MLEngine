# Strathmore Lost & Found — NLP matching engine

ICS 3202 · Deliverable 3 | Local presentation

This repository contains the Python matching engine, evaluated notebook, historical Logistic Regression baseline, and local authenticated backend. The companion [Android app](https://github.com/luiz-zer0/MAD_project/tree/codex/louis-deliverable3-demo) is maintained separately.

See also the [submission reproduction guide](docs/SUBMISSION.md).

## Start here

| Resource | Purpose |
|---|---|
| [NLP_Final.ipynb](notebooks/NLP_Final.ipynb) | Final assessed notebook: data preparation, model development, comparison, test scores and explanations. All 16 code cells have saved outputs. |
| [LogisticRegression_Baseline.ipynb](notebooks/LogisticRegression_Baseline.ipynb) | Original supervised baseline and historical Delhi experiment. |
| [Local presentation](docs/LOCAL_PRESENTATION.md) | One-command Windows startup and Android Studio emulator instructions. |
| [Data provenance](docs/data_provenance.md) | Source limitations, fictional labels and original baseline reproduction. |
| [Model evaluation](docs/model_evaluation.md) | Offline and operational ranking results. |
| [System guide](docs/system_guide.md) | Text retrieval, knowledge rules and preview API contract. |

## Run locally (no paid hosting)

Use Python 3.12, Node.js 22, Java 21 and Android Studio. Download dependencies and emulator binaries before presenting. From the repository root, first run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_presentation.ps1 -Install
```

On later runs omit `-Install`. Keep the emulator, API and worker windows open. Use the Android Studio emulator and build the companion Android branch with `-PdemoFirebase=true`. Sign in as `demo-student@strathmore.edu` or `demo-office@strathmore.edu` with local-only password `LocalDemo123!`. No Firebase cloud account or hosting credentials are needed.

| Local component | Address / command |
|---|---|
| Authenticated report API | http://127.0.0.1:8000/docs |
| Firebase Emulator UI | http://127.0.0.1:4000 |
| Auth / Firestore / Storage emulators | 9099 / 8080 / 9199 |
| Matching worker | `python -m service.worker` (started by the script) |
| Optional preview (not the mobile API) | `python -m uvicorn matching_system.api:app --host 127.0.0.1 --port 8001` |

The startup script creates fictional accounts and charger examples, checks actual API submissions and worker rankings, and rejects an invalid description. A clean local backend integration run is available [here](https://github.com/nimrod-kobia/SemProject_MLEngine/actions/runs/45504542376). Windows startup syntax was checked; the team still needs to rehearse the complete Android app on its laptop.

## How matching works

Lost report → text normalization → TF-IDF on active found item names/descriptions → cosine similarity → up to five positive-scoring candidates per lost-query call → category/location/date evidence → staff review.

The application uses `TfidfMatcher` and two knowledge-based agents. The agents explain agreement, conflict and missing facts after retrieval; they do not change the ranking. Scores are lexical similarity, not ownership probability. A worker saves results to local Firestore. Staff make the final decision.

Keyword overlap, structured attributes, Logistic Regression and hybrid methods are **evaluation comparisons only**. Keep them for the rubric; the mobile worker does not run these alternatives. No CLIP or image inference is used.

## Data and verified results

Office inventory records describe 687 reported quantities (388 June–July and 299 August), not matched report pairs. The separate controlled fictional benchmark contains 24 found reports, 27 lost queries and 192 labelled pairs (24 positive, 168 negative). Training, validation and test use distinct related item groups. Independent label review remains necessary.

| Method | Top-1 | Recall@5 | MRR | False suggestions |
|---|---:|---:|---:|---:|
| Keyword | 0.250 | 0.875 | 0.541 | 1/2 |
| Attributes | 1.000 | 1.000 | 1.000 | 1/2 |
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1/2 |
| Offline TF-IDF | 0.250 | 0.500 | 0.391 | 2/2 |
| Hybrid | 1.000 | 1.000 | 1.000 | 2/2 |

Operational TF-IDF (same ranker as the app, evaluated separately): Top-1 4/8 (50%), Recall@5 7/8 (87.5%), MRR@5 0.629 and suggestions on both unmatched queries. These are controlled results, not campus accuracy. TF-IDF is retained for lightweight inventory-based free-text ranking, not because it won the offline comparison.

## Reproduce the notebook

```powershell
python -m pip install -r requirements.txt
python -m pip install nbconvert ipykernel
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/NLP_Final.ipynb --ExecutePreprocessor.timeout=180
```

The original clean-kernel execution [passed](https://github.com/nimrod-kobia/SemProject_MLEngine/actions/runs/45503554200). The optional HTTP preview cell is skipped by default. To regenerate the benchmark separately:

```powershell
python -m scripts.prepare_ranking_benchmark
python -m scripts.validate_ranking_dataset data/processed/ranking
python -m scripts.evaluate_ranking_benchmark
```

## Repository layout

```text
notebooks/       Final NLP and historical base notebooks
matching_system/ TF-IDF, knowledge agents and baseline feature/model code
service/         Authenticated FastAPI, Firebase runtime and worker
firebase/        Emulator configuration, rules and tests
data/            Current fictional fixtures, office aggregates and baseline data
scripts/         Local startup, dataset preparation and evaluation
tests/           Model, data, API and emulator integration tests
results/         Current benchmark and historical baseline evidence
docs/            Current guides and provenance
```

## Preserve the baseline

The historical Delhi found-only dataset has 13,713 records; the raw CSV and sensitive row-level derivatives are deliberately Git-ignored. Obtain the source and verify its hash using [data provenance](docs/data_provenance.md), run `python -m scripts.create_delhi_experiment`, then open the base notebook. Its 120 fictional claims and 360 constructed pairs are not genuine matches. The older campus classifier remains reproducible with `python -m scripts.run_logistic_regression_experiment`. Keep all baseline data, scripts, figures and tests.

## Verification and boundaries

See [Firebase setup](FIREBASE_SETUP.md) and [final checks](docs/deliverable3_final_checks.md). Local emulators are temporary and intended only for fictional data. The presentation does not require Render, Cloudflare, Docker or a deployed Firebase project. Git history retains retired deployment experiments; do not deploy them or use production credentials for the presentation. Repository changes do not modify any live Firebase account.

[Team contributions](docs/contributions.md) · [Documentation index](docs/README.md)

## Report relevance and zero-result handling

A conservative rule-based NLP guard rejects clearly off-topic standalone sentences (for example, "I had chicken for lunch") and obvious placeholder input before a new report/job is saved. It deliberately allows uncertain descriptions and items absent from the inventory. It is not a trained semantic classifier and does not guarantee detection of arbitrary irrelevant text. Valid reports with no positive lexical candidates complete normally: the matches endpoint returns `matching_outcome: no_matches`, `match_count: 0` and an empty list. Pending and failed jobs are distinct. Tests cover rejection, correction, completed zero matches and ordinary retrieval. No baseline, TF-IDF score or benchmark was changed.

## Live alerts and claim review

The worker writes one durable notification for each newly suggested lost/found pair, including when staff check in an item after a student reported it lost. Retrying the same pair does not reset read status or create a duplicate alert. GET /v1/notifications returns the newest 100 account-scoped alerts with IDs and unread count; POST /v1/notifications/{id}/read marks one read. GET /v1/claims/{id} provides current claim state and real evidence for authorized staff, with private found-item details hidden from students and technical administrators. Claim approval and physical collection remain staff-controlled. Local alerts are in-app; background FCM push is not available in emulator mode. See tests/test_firebase_integration.py for the lost-first, found-later, review and handover test.
