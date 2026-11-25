# Ranking benchmark version 2: harder fictional cases

Version 1 remains frozen in `data/processed/ranking/` and `results/ranking_benchmark.json`. Its lost and found records shared structured fields by construction, so attribute, Logistic Regression and hybrid Top-1 scores reached 1.00. The version 1 test cases were inspected before version 2 was created and are used only for version 2 development.

## Design and provenance

The new 12-group test fixture was written and frozen before running the version 2 evaluator. Its SHA-256 is recorded in `data/processed/ranking_v2/provenance.json` and checked by `tests/test_ranking_v2.py`. The 24 older groups are assigned to training (16) and validation (8), with role-specific metadata variation. The 12 new test groups contain pairs of similar bottles, chargers, pencil pouches, flasks, glasses cases and USB cables. Lost descriptions sometimes omit colour or brand, use a different colour word, or name a different campus location. Four new no-match queries resemble existing categories. These are **AI-authored controlled fiction**, not actual student losses, Office records or independently verified item identities. The 2026 Office inventory supplied only aggregate category quantities.

The design uses text and metadata as separate evidence because published lost-object retrieval work combines textual search with geographic and colour information ([Abu Dhabi University research](https://repository.adu.ac.ae/items/50e44846-18b5-4d8b-a629-e9e85abec7c7)); that source does not supply the records or performance values used here. The specific cases and wording were constructed for this project. All 36 found records form one fixed pool. Item-group splits are 16 train, 8 validation and 12 test. TF-IDF vocabulary and Logistic Regression are fitted on training data. The hybrid weight and suggestion cutoff use validation data only. The frozen test was evaluated once.

## Measured held-out results

Each method faced 12 matched queries and four no-match queries against 36 found candidates. “Strong suggestion” uses that method's validation-selected cutoff; it is **not** an ownership probability.

| Method | Top-1 | Recall@5 | MRR | Correct item above cutoff | False strong suggestions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Keyword overlap | 0.500 | 0.917 | 0.642 | 0/12 | 1/4 |
| Structured attributes | 0.167 | 1.000 | 0.443 | 10/12 | 3/4 |
| Logistic Regression pair baseline | 0.250 | 1.000 | 0.549 | 12/12 | 4/4 |
| TF-IDF cosine | 0.167 | 0.750 | 0.375 | 1/12 | 2/4 |
| Hybrid (TF-IDF weight 0.25) | 0.583 | 1.000 | 0.764 | 10/12 | 4/4 |

The hybrid ranks seven of twelve controlled positives first and all twelve within five, but it flags every no-match test query as strong. It must **not** be presented as a reliable no-match detector. Keyword retrieval has fewer false suggestions at its cutoff, but it identifies none of the matched test queries as strong. The thresholds reflect only three validation no-match examples and are not calibrated. The very small synthetic holdout gives unstable estimates; 1/12 changes Top-1 by 0.083. The same-category distractors expose ranking uncertainty, while the no-match cases expose unsafe confidence.

## Reproduction and limits

```powershell
python -m scripts.prepare_ranking_benchmark_v2
python -m scripts.validate_ranking_dataset data/processed/ranking_v2
python -m scripts.evaluate_ranking_benchmark --data data/processed/ranking_v2 --result results/ranking_benchmark_v2.json
python -m scripts.build_nlp_notebook
python -m scripts.execute_nlp_notebook_inprocess
python -m pytest -q tests/test_ranking_v2.py tests/test_ranking_benchmark.py tests/test_ranking_dataset_validator.py
```

`notebooks/NLP_Final.ipynb` now presents version 2 as the primary controlled exercise. Its checked-in outputs were executed in one Python process; a separate clean Jupyter kernel run passed on 25 November 2026. No real campus performance, ownership decision, authenticated HTTP or Android end-to-end result is inferred from this benchmark. Independent student review of descriptions and labels, source-inventory verification and a new real-data holdout remain required for field claims. Do not revise this test set in response to its scores; create a new version if the evaluation design changes.
