import pandas as pd

df = pd.read_csv("reports/test_predictions.csv")

df["bucket"] = pd.qcut(df["prediction"], 5, labels=False)

print("\nRetorno médio por bucket:")
print(df.groupby("bucket")["target_5d"].mean())

print("\nContagem por bucket:")
print(df["bucket"].value_counts().sort_index())