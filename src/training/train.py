import argparse
import json
from pathlib import Path
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from src.core.logging import get_logger
from src.core.paths import DATA_DIR
from src.training.datasets import load_features, make_xy
from src.training.cv import walk_forward_splits, dates_to_indices
from src.training.models.baseline import RidgeBaseline

log = get_logger("train")

def models_dir(market: str) -> Path:
    return DATA_DIR / "models" / market.upper()

def train_market(market: str):
    df = load_features(market)
    X, y, meta, feat_cols = make_xy(df)

    splits = walk_forward_splits(meta["date"], n_splits=5, test_size_days=63)
    if not splits:
        raise RuntimeError("Sem splits suficientes para walk-forward. Aumente histórico ou diminua test_size_days.")

    rmses = []
    model_cfg = RidgeBaseline(alpha=1.0)
    for k, (tr_dates, te_dates) in enumerate(splits, start=1):
        tr_idx, te_idx = dates_to_indices(meta["date"], tr_dates, te_dates)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model_cfg.build()),
        ])

        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[te_idx])
        rmse = float(np.sqrt(mean_squared_error(y.iloc[te_idx], pred)))
        rmses.append(rmse)
        log.info(f"[{market}] split {k}/{len(splits)} RMSE={rmse:.6f} | test_n={len(te_idx)}")

    metrics = {
        "market": market.upper(),
        "model": "ridge_baseline",
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "rmse_splits": rmses,
        "n_rows": int(len(df)),
        "n_features": int(len(feat_cols)),
    }

    # Treina final com tudo (até a última data disponível)
    final_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", model_cfg.build()),
    ])
    final_pipe.fit(X, y)

    out_dir = models_dir(market)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_pipe, out_dir / "ridge_baseline.joblib")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"[{market}] salvo modelo em {out_dir/'ridge_baseline.joblib'}")
    log.info(f"[{market}] métricas: rmse_mean={metrics['rmse_mean']:.6f} rmse_std={metrics['rmse_std']:.6f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True, choices=["US","BR"])
    args = p.parse_args()
    train_market(args.market)

if __name__ == "__main__":
    main()
