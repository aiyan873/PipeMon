import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Configuring
FILE_IN = "/Users/User/Dataset_raw.csv"
FILE_OUT = "pipe_vibration_dataset_with_features.csv"
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
    mu = df[col].mean()
    sd = df[col].std(ddof=1)
    mask &= (df[col] >= mu - SIGMA*sd) & (df[col] <= mu + SIGMA*sd)

df = df[mask].reset_index(drop=True)

# Timestamps reset and RMS col addition
df["Timestamps"] = np.arange(0, len(df) * 100, 100)
df["rms"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

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

# Global features
features = {}
for col in ["x", "y", "z", "rms"]:
    features[f"{col}_mean"] = df[col].mean()
    features[f"{col}_std"] = df[col].std(ddof=1)
    features[f"{col}_skew"] = safe_skew(df[col])
    features[f"{col}_kurt"] = safe_kurt(df[col])

# Broadcasting global features as columns
for k, v in features.items():
    df[k] = v

# Renaming Columns
df.rename(columns={
    "Timestamps": "timestamp",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "rms": "RMS",
    "x_mean": "X_mean",
    "y_mean": "Y_mean",
    "z_mean": "Z_mean",
    "rms_mean": "RMS_mean",
    "x_std": "X_std",
    "y_std": "Y_std",
    "z_std": "Z_std",
    "rms_std": "RMS_std",
    "x_skew": "X_skew",
    "y_skew": "Y_skew",
    "z_skew": "Z_skew",
    "rms_skew": "RMS_skew",
    "x_kurt": "X_kurt",
    "y_kurt": "Y_kurt",
    "z_kurt": "Z_kurt",
    "rms_kurt": "RMS_kurt",
    "Target": "target"
}, inplace=True)

# Keep only required columns in exact order
final_columns = [
    "timestamp", "X", "Y", "Z", "RMS",
    "X_mean", "Y_mean", "Z_mean", "RMS_mean",
    "X_std", "Y_std", "Z_std", "RMS_std",
    "X_skew", "Y_skew", "Z_skew", "RMS_skew",
    "X_kurt", "Y_kurt", "Z_kurt", "RMS_kurt",
    "target"
]

df = df[final_columns]

# Save
df.to_csv(FILE_OUT, index=False)

