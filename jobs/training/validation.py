from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = BASE_DIR / "artifacts" / "reports"
PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
OUTPUT_PATH = REPORT_DIR / "google_sheets_validation_trades.csv"

TOP_N = 5
BOTTOM_N = 5
REBALANCE_EVERY_N_DAYS = 5


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {PREDICTIONS_PATH}. Rode antes o train_model.py."
        )

    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["trade_date"])

    required_cols = {"trade_date", "symbol", "target_5d", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em test_predictions.csv: {sorted(missing)}")

    return df


def select_rebalance_dates(preds: pd.DataFrame, step: int) -> list[pd.Timestamp]:
    unique_dates = sorted(preds["trade_date"].drop_duplicates())
    return unique_dates[::step]


def build_validation_trades(
    preds: pd.DataFrame,
    top_n: int,
    bottom_n: int,
    rebalance_every_n_days: int,
) -> pd.DataFrame:
    preds = preds.sort_values(["trade_date", "prediction"], ascending=[True, False]).copy()
    rebalance_dates = select_rebalance_dates(preds, step=rebalance_every_n_days)

    rows: list[dict] = []

    for entry_date in rebalance_dates:
        day_df = preds[preds["trade_date"] == entry_date].sort_values(
            "prediction",
            ascending=False,
        ).copy()

        longs = day_df.head(top_n).copy()
        longs["side"] = "long"

        shorts = day_df.tail(bottom_n).copy()
        shorts["side"] = "short"

        positions = pd.concat([longs, shorts], ignore_index=True)

        for _, row in positions.iterrows():
            symbol = str(row["symbol"])
            rows.append(
                {
                    "entry_date": pd.to_datetime(entry_date).date().isoformat(),
                    "exit_date_calendar": (
                        pd.to_datetime(entry_date) + pd.Timedelta(days=7)
                    ).date().isoformat(),
                    "holding_days": 5,
                    "symbol": symbol,
                    "google_ticker": f"BVMF:{symbol}",
                    "side": row["side"],
                    "weight": 1.0 / (top_n + bottom_n),
                    "prediction": float(row["prediction"]),
                    "target_5d_model": float(row["target_5d"]),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["entry_date", "side", "prediction"], ascending=[True, True, False])
    return out


def main() -> None:
    preds = load_predictions()
    trades = build_validation_trades(
        preds=preds,
        top_n=TOP_N,
        bottom_n=BOTTOM_N,
        rebalance_every_n_days=REBALANCE_EVERY_N_DAYS,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUTPUT_PATH, index=False)

    print(f"Arquivo salvo em: {OUTPUT_PATH}")
    print("\nPrimeiras linhas:")
    print(trades.head(20).to_string(index=False))


if __name__ == "__main__":
    main()