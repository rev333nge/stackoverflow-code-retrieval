"""Train a LambdaMART reranker on the Phase 5 retrieval features.

    python scripts/train_ltr.py <variant>
      variant = a | b

Loads data/processed/ltr_train_<variant>.parquet and ltr_val_<variant>.parquet
(built by build_ltr_features.py), trains a LightGBM lambdarank model with
early stopping on val NDCG@10, and saves it to data/index/ltr_<variant>/model.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from evaluate import PROCESSED

INDEX_ROOT = Path("data/index")

FEATURES = ["bm25_score", "bm25_rank", "dense_score", "dense_rank", "rrf_score"]

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.1,      # chosen via tune_ltr.py val search on the python corpus -
    "num_leaves": 15,          # best of 12 configs (val ndcg@10 0.8387, best_iter=290).
    "min_data_in_leaf": 100,   # Different winner than the pandas/numpy run (lr=0.2) -
    "verbosity": -1,           # re-tuning on the bigger corpus actually mattered here.
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30


def load_dataset(path: Path) -> lgb.Dataset:
    df = pd.read_parquet(path)
    groups = df.groupby("query_id", sort=False).size().values
    return lgb.Dataset(df[FEATURES], label=df["label"], group=groups, free_raw_data=False)


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"

    train_path = PROCESSED / f"ltr_train_{variant}.parquet"
    val_path = PROCESSED / f"ltr_val_{variant}.parquet"
    for p in (train_path, val_path):
        if not p.exists():
            raise SystemExit(f"missing {p} - run: python scripts/build_ltr_features.py {variant} <split>")

    print("loading features ...")
    train_set = load_dataset(train_path)
    val_set = load_dataset(val_path)

    print("training ...")
    booster = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=10),
        ],
    )

    out_dir = INDEX_ROOT / f"ltr_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.txt"
    booster.save_model(str(model_path))

    print(f"\nbest iteration     : {booster.best_iteration}")
    print(f"best val ndcg@10   : {booster.best_score['val']['ndcg@10']:.4f}")
    print(f"wrote {model_path}")

    importances = sorted(
        zip(FEATURES, booster.feature_importance(importance_type="gain")),
        key=lambda x: -x[1],
    )
    print("\nfeature importance (gain):")
    for name, imp in importances:
        print(f"  {name:12s} {imp:,.0f}")


if __name__ == "__main__":
    main()
