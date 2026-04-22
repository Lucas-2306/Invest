import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = BASE_DIR / "artifacts" / "reports"

df = pd.read_csv(REPORT_DIR / "test_predictions.csv")

df["bucket"] = pd.qcut(df["prediction"], 5, labels=False)

print("\nRetorno médio por bucket (target de treino):")
print(df.groupby("bucket")["target_model"].mean())

print("\nRetorno médio por bucket (execução t+1):")
print(df.groupby("bucket")["target_exec_t1"].mean())

print("\nContagem por bucket:")
print(df["bucket"].value_counts().sort_index())