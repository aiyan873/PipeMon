import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis



# Configuring
FILE_IN = "/Users/User/Dataset_raw.csv"
FILE_OUT = "Dataset_cleaned_with_global_features.csv"
SIGMA = 3

df = pd.read_csv(FILE_IN)




# Handling missing data
fill_values = {}
for col in df.columns:
    if col == "Target":
        fill_values[col] = df[col].mode(dropna=True)[0]
    elif pd.api.types.is_numeric_dtype(df[col]):
        fill_values[col] = df[col].median()
    else:
        fill_values[col] = df[col].mode(dropna=True)[0]

df = df.fillna(value=fill_values)




# Removing outliers
cols_for_outliers = ["x", "y", "z"]

mask = pd.Series(True, index=df.index)
for col in cols_for_outliers:
    mu  = df[col].mean()
    sd  = df[col].std(ddof=1)
    mask &= (df[col] >= mu - SIGMA*sd) & (df[col] <= mu + SIGMA*sd)

df = df[mask].reset_index(drop=True)



# Timestamps reset and RMS col addition

df["Timestamps"] = np.arange(0, len(df) * 100, 100)

df["rms"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)





# Features Extraction

# Safe versions of skew/kurt that return 0 if variance ~ 0.
def safe_skew(series: pd.Series) -> float:
    s = series.to_numpy(dtype=float)
    if np.std(s, ddof=1) < 1e-12:
        return 0.0
    return float(skew(s, bias=False))

def safe_kurt(series: pd.Series) -> float:
    s = series.to_numpy(dtype=float)
    if np.std(s, ddof=1) < 1e-12:
        return 0.0
    return float(kurtosis(s, fisher=True, bias=False))

features = {}
for col in ["x", "y", "z", "rms"]:
    features[f"{col}_mean"] = df[col].mean()
    features[f"{col}_std"]  = df[col].std(ddof=1)
    features[f"{col}_skew"] = safe_skew(df[col])
    features[f"{col}_kurt"] = safe_kurt(df[col])





# Broadcasting global features as columns

for k, v in features.items():
    df[k] = v




# Reordering Columns

feature_cols_order = [
    "x_mean","y_mean","z_mean","rms_mean",
    "x_std","y_std","z_std","rms_std",
    "x_skew","y_skew","z_skew","rms_skew",
    "x_kurt","y_kurt","z_kurt","rms_kurt"
]

feature_cols_order = [c for c in feature_cols_order if c in df.columns]

base_cols = ["Timestamps","x","y","z","rms"]
end_cols  = ["Target"] if "Target" in df.columns else []

new_order = base_cols + feature_cols_order + end_cols
other_cols = [c for c in df.columns if c not in new_order]
df = df[new_order + other_cols]


df.to_csv(FILE_OUT, index=False)

