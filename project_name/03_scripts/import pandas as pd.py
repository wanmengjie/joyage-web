import pandas as pd
p = r"C:\Users\lenovo\Desktop\20250906 charls  klosa\project_name\02_processed_data\v2025-10-01\frozen\charls_mean_mode_Xy.csv"
df = pd.read_csv(p)
print(df["depression_bin"].value_counts(dropna=False))
