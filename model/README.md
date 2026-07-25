# Fraud Feature And Model Pipeline

The production training input is the cleaned upstream dataset
`transactions_cleaned.parquet`. Feature definitions and state semantics live only in
`fraud_pipeline/features.py`.

Prepare chronological feature artifacts:

```powershell
python -m model.prepare_features `
  --data-path data/processed/transactions_cleaned.parquet `
  --artifacts-dir model/artifacts `
  --feature-config deployment_safe
```

Train, compare, select, and verify all supported models:

```powershell
python -m model.train_models `
  --artifacts-dir model/artifacts `
  --model-types all `
  --false-alarm-cost 5
```

For a smoke run, add `--limit` to feature preparation and use `--quick` during
training. `model/train_model.py` remains a compatibility wrapper that invokes these
two stages; it no longer contains feature or training implementations.

Full Parquet preparation streams complete-step batches (`--chunk-size 250000` by
default), so dynamic state crosses batches without allowing same-step visibility.
The cleaned Parquet must be ordered by `step`; use `--in-memory` only for a small,
unsorted compatibility input. `full_paysim` is an analysis configuration containing
post-transaction balances. Production export intentionally requires
`deployment_safe`.

The split manifest is chronological and step-atomic: the first 60% of rows by
complete PaySim steps are train, the next 20% validation, and the final 20% test.
Thresholds and model selection use validation business cost only. The held-out test
split is evaluated once after model and hybrid thresholds are frozen.

The deployable output is the atomic bundle
`model/artifacts/fraud_pipeline_selected.joblib`. It contains the fitted
preprocessor, model, feature order, ML/hybrid thresholds, score weights, data/split
versions, and cost configuration. `model/selected_model.json` is only a lightweight
pointer to that bundle.

Docker Compose sets `REQUIRE_ML_ARTIFACT=true`. Run both stages before starting the
Spark application; a missing/incompatible artifact is a startup error rather than a
silent hybrid-to-rule-only downgrade. Atomic replacements are refreshed at a Spark
microbatch boundary.
