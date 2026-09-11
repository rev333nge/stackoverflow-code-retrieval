"""Search LightGBM hyperparameters for the LambdaMART reranker, on val.

    python scripts/tune_ltr.py <variant>
      variant = a | b

Manual random search: sample N configs from the space below, train each
once with early stopping (same budget/patience as the real model), score
against the real held-out val split, keep the best - same discipline as
the Phase 5 k sweep. The winner is what train_ltr.py's PARAMS should use.
"""

from __future__ import annotations

import random
import sys
import time

import lightgbm as lgb
import pandas as pd

from evaluate import PROCESSED

FEATURES = ["bm25_score", "bm25_rank", "dense_score", "dense_rank", "rrf_score"]

SEARCH_SPACE = {
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63, 127],
    "min_data_in_leaf": [20, 50, 100],
}

N_SAMPLES = 12
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30
SEED = 0


def load_dataset(path, features) -> lgb.Dataset:
    df = pd.read_parquet(path)
    groups = df.groupby("query_id", sort=False).size().values
    return lgb.Dataset(df[features], label=df["label"], group=groups, free_raw_data=False)


def sample_configs(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen = set()
    configs = []
    while len(configs) < n:
        cfg = tuple(rng.choice(v) for v in SEARCH_SPACE.values())
        if cfg in seen:
            continue
        seen.add(cfg)
        configs.append(dict(zip(SEARCH_SPACE.keys(), cfg)))
    return configs


def run_config(cfg: dict, train_set, val_set):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "verbosity": -1,
        **cfg,
    }
    t0 = time.time()
    booster = lgb.train(
        params, train_set, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    elapsed = time.time() - t0
    return booster.best_score["val"]["ndcg@10"], booster.best_iteration, elapsed


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    train_path = PROCESSED / f"ltr_train_{variant}.parquet"
    val_path = PROCESSED / f"ltr_val_{variant}.parquet"

    print("loading features ...")
    train_set = load_dataset(train_path, FEATURES)
    val_set = load_dataset(val_path, FEATURES)

    configs = sample_configs(N_SAMPLES, SEED)
    print(f"\n=== searching {len(configs)} configs ({NUM_BOOST_ROUND} rounds max, patience {EARLY_STOPPING_ROUNDS}) ===")
    results = []
    for i, cfg in enumerate(configs, 1):
        ndcg, best_iter, elapsed = run_config(cfg, train_set, val_set)
        results.append((ndcg, best_iter, elapsed, cfg))
        print(f"[{i:2d}/{len(configs)}] ndcg@10={ndcg:.4f}  best_iter={best_iter:4d}  {elapsed:5.1f}s  {cfg}")

    results.sort(key=lambda x: -x[0])
    best_ndcg, best_iter, _, best_cfg = results[0]
    print(f"\nbest config : {best_cfg}")
    print(f"best iter   : {best_iter}")
    print(f"val ndcg@10 : {best_ndcg:.4f}")


if __name__ == "__main__":
    main()
