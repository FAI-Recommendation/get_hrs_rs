import sys
import os
sys.path.append(r"e:\DoCode\CD2\source\Source\get_hrs_rs\data_evaluate")
from load_data import load_from_csv

csv_path = r"e:\DoCode\CD2\source\Source\get_hrs_rs\data_evaluate\data_wandb\all_runs_summary.csv"
df = load_from_csv(csv_path)

print("COLUMNS:")
print(list(df.columns))
print("\nDATA:")
print(df.to_string())
