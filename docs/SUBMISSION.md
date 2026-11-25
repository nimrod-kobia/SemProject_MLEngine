# Submission guide

## Start here

Open the repository root in VS Code and select a Python environment with
`python -m pip install -r requirements-dev.txt` installed. Use Python 3.11 or newer.

The primary submission notebook is [NLP_Final.ipynb](../notebooks/NLP_Final.ipynb).
It locates the project from either the root or the notebooks directory.
The earlier [Logistic Regression notebook](../notebooks/LogisticRegression_Baseline.ipynb)
is retained for the CAT 2 baseline and comparison; it is not the mobile worker.

## Reproduce the assessment

Run these commands from the repository root:

```powershell
python -m scripts.prepare_ranking_benchmark_v2
python -m scripts.validate_ranking_dataset data/processed/ranking_v2
python -m pytest tests -q
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/NLP_Final.ipynb
```

The final notebook retains Nimrod's latest workflow. The optional version 2 scripts regenerate a separate fictional comparison and its results. The frozen test
fixture is unchanged; its provenance hash normalizes Windows line endings.
Do not edit the fixture to improve scores. Review the examples and labels as a team.

## Which API to demonstrate

- `matching_system.api:app`: standalone synthetic preview; no accounts or persistence.
- `service.api:app`: authenticated mobile backend.
- `python -m service.worker`: separate durable matching worker.

Use [Firebase setup](../FIREBASE_SETUP.md) for the complete local service workflow.
The offline hybrid comparison is not the operational TF-IDF worker. Similarity is
not ownership probability, and staff must verify claims before handover.

## Repository organization

`matching_system/` contains retrieval, knowledge agents and the retained classifier;
`service/` contains authentication, persistence and background processing;
`scripts/` contains reproducible preparation and evaluation commands; `tests/`
contains regression checks. Older fixtures, results and handoffs preserve the
coursework progression and are retained for comparison rather than deleted.

## Submission limitations

The data is controlled fiction and cannot establish real campus accuracy. The
hybrid comparison produces false strong suggestions for all four no-match test
queries. Production deployment and a current device walkthrough require separate
verification. Firebase integration tests intentionally skip unless local emulators
and their explicit environment flag are configured; a skip is not a pass.
