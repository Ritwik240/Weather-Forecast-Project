"-------Blockchain for Data Integrity--------"
"----Definitions----"

import hashlib
import json
from datetime import datetime
import requests
import sys
import pandas as pd

# ---------------------------------
# 1. THE BLOCKCHAIN BLUEPRINT
# ---------------------------------
class Block:
    def __init__(self, data, previous_hash):
        self.timestamp = str(datetime.now())
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(data={"message": "Genesis Block"}, previous_hash="0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_data):
        latest_block = self.get_latest_block()
        new_block = Block(data=new_data, previous_hash=latest_block.hash)
        self.chain.append(new_block)
        return new_block

# ---------------------------------
# 2. THE FILE HASHER TOOL
# ---------------------------------
def hash_url_file(url):
    """Calculates the SHA-256 hash of a file streamed from a URL."""
    h = hashlib.sha256()
    print(f"Hashing file from: {url}")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                h.update(chunk)
        print("Hash calculation complete.")
        return h.hexdigest()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading or hashing file: {e}")
        return None
    
"---Calculating Hash Value---"

# 1. PASTE YOUR GITHUB "RAW" LINK HERE
raw_dataset_url = "https://raw.githubusercontent.com/Ritwik240/Weather-Dataset/refs/heads/main/Unified_Weather_Dataset_Latest.json"

# 2. Calculate and print the hash
official_hash = hash_url_file(raw_dataset_url)

if official_hash:
    print("\n--- COPY THIS HASH ---")
    print(official_hash)
    print("------------------------")

"---Hash Checking---"

import pandas as pd
import numpy as np
import hashlib
import requests
import sys
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

# =======================================================
# 1. CONFIGURATION
# =======================================================
raw_dataset_url = "https://raw.githubusercontent.com/Ritwik240/Weather-Dataset/refs/heads/main/Unified_Weather_Dataset_Latest.json"
OFFICIAL_DATASET_HASH = "e1603c17297d93b13aaeef0892f9f9704085dff7e183edbf11bf9e94b60e5e65"

# =======================================================
# 2. HASH CHECK FUNCTION
# =======================================================
def hash_url_file(url):
    h = hashlib.sha256()
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                h.update(chunk)
        return h.hexdigest()
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not download or hash file: {e}")
        return None

# =======================================================
# 3. STEP 1 — DATA INTEGRITY CHECK
# =======================================================
print("--- [STEP 1] Running Data Integrity Check ---")

current_hash = hash_url_file(raw_dataset_url)

if not current_hash:
    print("Hash calculation failed. Aborting.")
    sys.exit(1)

if current_hash != OFFICIAL_DATASET_HASH:
    print("\n❌ DANGER! DATA TAMPERING DETECTED!")
    sys.exit(1)
else:
    print("\n✅ SUCCESS: Data integrity verified.")
    print("--- Proceeding to anomaly detection ---")

# =======================================================
# 4. LOAD DATA
# =======================================================
print("\nLoading dataset...")

try:
    df_combined = pd.read_json(raw_dataset_url, orient="records", convert_dates=["Date"])
    df_combined = df_combined.sort_values("Date").reset_index(drop=True)
    print(f"Dataset loaded. Shape: {df_combined.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Identify numerical columns
numerical_cols = df_combined.select_dtypes(include=np.number).columns

if len(numerical_cols) == 0:
    print("No numerical columns found.")
    sys.exit(1)

# Copy numerical data
df_numerical = df_combined[numerical_cols].copy()

# Fill missing values
if df_numerical.isnull().values.any():
    print("Filling missing values...")
    df_numerical = df_numerical.fillna(df_numerical.mean())

# =======================================================
# 5. PARAMETER-WISE ISOLATION FOREST
# =======================================================
print("\nRunning per-feature IsolationForest...")

for col in numerical_cols:
    iso = IsolationForest(contamination="auto", random_state=42)
    flags = iso.fit_predict(df_numerical[[col]])
    df_combined[col + "_anomaly"] = pd.Series(flags).map({1: 0, -1: 1})

print("Added per-feature anomaly flags.")

# =======================================================
# 6. Z-SCORE SEVERITY
# =======================================================
print("Computing z-score severity...")

z_scores = df_numerical.apply(zscore).fillna(0)

for col in numerical_cols:
    df_combined[col + "_severity"] = (
        df_combined[col + "_anomaly"] * z_scores[col].abs()
    ).fillna(0)

# =======================================================
# 7. ROW ANOMALY SCORE
# =======================================================
severity_cols = [col + "_severity" for col in numerical_cols]
df_combined["row_anomaly_score"] = df_combined[severity_cols].sum(axis=1)

# =======================================================
# 8. ANOMALY CATEGORY LABELS
# =======================================================
print("Generating anomaly categories...")

def build_anomaly_category(row):
    anomaly_cols = [col for col in numerical_cols if row[col + "_anomaly"] == 1]

    if len(anomaly_cols) == 0:
        return "No anomaly"

    # Convert into readable labels
    readable = [col.replace("_", " ").title() for col in anomaly_cols]

    if len(readable) == 1:
        return f"{readable[0]} anomaly"

    return " + ".join(readable) + " anomalies"

df_combined["anomaly_category"] = df_combined.apply(build_anomaly_category, axis=1)

# =======================================================
# 9. SEVERITY LEVEL BUCKET
# =======================================================
print("Adding severity levels...")

def severity_label(score):
    if score == 0:
        return "None"
    elif score < 1:
        return "Low"
    elif score < 3:
        return "Moderate"
    else:
        return "High"

df_combined["severity_level"] = df_combined["row_anomaly_score"].apply(severity_label)

# =======================================================
# 10. REPORT
# =======================================================
print("\n==================== FINAL REPORT ====================")
print("Total rows:", len(df_combined))

print("\nSeverity distribution:")
print(df_combined["severity_level"].value_counts())

print("\nExample anomalies:")
print(df_combined[["row_anomaly_score", "anomaly_category", "severity_level"]].nlargest(5, "row_anomaly_score"))

print("\nAll operations completed successfully.")

"-----Model Training-----"

# forecasting_script_multi_target.py
# Loads Unified_Weather_Dataset_Latest.json from GitHub, trains multi-output stacking models
# for all targets (Temperature_C, Humidity_%, UV_Index, WindSpeed_m/s, Rainfall_mm),
# optionally trains a Rain classifier, and creates a 7-day forecast JSON.

import pandas as pd
import numpy as np
import joblib
import sys
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier

# -----------------------
# CONFIG
# -----------------------
RAW_JSON = "https://raw.githubusercontent.com/Ritwik240/Weather-Dataset/refs/heads/main/Unified_Weather_Dataset_Latest.json"
N_FORECAST_DAYS = 7
TARGETS = ["Temperature_C", "Humidity_%", "UV_Index", "WindSpeed_m/s", "Rainfall_mm"]
RAIN_CLASS_COL = "RainToday"
RAIN_THRESHOLD_MM = 2.5

# -----------------------
# 0. Load dataset (from GitHub raw JSON)
# -----------------------
print("Loading dataset from:", RAW_JSON)
try:
    # read_json can accept URL; ensure Date parsed
    df = pd.read_json(RAW_JSON, orient="records", convert_dates=["Date"])
    print("Loaded rows:", len(df), "columns:", list(df.columns))
except Exception as e:
    print("Error loading dataset:", e)
    sys.exit(1)

# Ensure Date exists and is datetime
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

# -----------------------
# 1. Ensure anomaly & severity columns exist (if data_generation.py created them, we use them;
#    otherwise we create zero placeholders so model features stay consistent)
# -----------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

expected_anom_cols = []
expected_sev_cols = []
for col in numeric_cols:
    an = f"{col}_anomaly"
    sv = f"{col}_severity"
    expected_anom_cols.append(an)
    expected_sev_cols.append(sv)
    if an not in df.columns:
        df[an] = 0
    if sv not in df.columns:
        df[sv] = 0.0

# Create RainToday if absent
if RAIN_CLASS_COL not in df.columns:
    if "Rainfall_mm" in df.columns:
        df[RAIN_CLASS_COL] = (df["Rainfall_mm"] > RAIN_THRESHOLD_MM).astype(int)
    else:
        df[RAIN_CLASS_COL] = 0


# =========================================================
# ✨ NEW SECTION: Create Time-Series Features
# =========================================================
print("Engineering time-series features (lags and cyclical)...")

# --- A. Cyclical (Time of Year) Features ---
# This helps the model understand seasonality
df['day_of_year'] = df['Date'].dt.dayofyear
df['month'] = df['Date'].dt.month

# Encode them as cycles (so 365 is close to 1)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# --- B. Lag (Memory) Features ---
# This gives the model "memory" of recent conditions
lag_days = [1, 3, 7] # 1 day ago, 3 days ago, 1 week ago
lag_cols = ["Temperature_C", "Humidity_%", "Rainfall_mm", "WindSpeed_m/s"]

for col in lag_cols:
    for day in lag_days:
        df[f"{col}_lag_{day}"] = df[col].shift(day)

# --- C. Rolling (Trend) Features ---
# This helps the model see recent trends
rolling_windows = [3, 7] # 3-day and 7-day windows
rolling_cols = ["Temperature_C", "Rainfall_mm"]

for col in rolling_cols:
    for window in rolling_windows:
        df[f"{col}_rolling_mean_{window}d"] = df[col].rolling(window=window).mean()
        df[f"{col}_rolling_sum_{window}d"] = df[col].rolling(window=window).sum()

# Drop rows with NaNs created by lags/rolling (important!)
df = df.dropna().reset_index(drop=True)
print(f"Dataset shape after adding features: {df.shape}")

# -----------------------
# 2. Build feature columns
# -----------------------
# Use numeric columns excluding multi-output targets...

# -----------------------
# 2. Build feature columns
# -----------------------
# Use numeric columns excluding multi-output targets, plus anomaly & severity columns
base_numeric_cols = [c for c in numeric_cols if c not in TARGETS]
feature_cols = list(base_numeric_cols) + sorted(expected_anom_cols) + sorted(expected_sev_cols)
# keep only existing and unique
seen = set()
feature_cols = [c for c in feature_cols if c in df.columns and not (c in seen or seen.add(c))]

if len(feature_cols) == 0:
    print("No features available to train. Aborting.")
    sys.exit(1)

print("Feature columns count:", len(feature_cols))

# -----------------------
# Helper: create supervised multi-output target vectors for n days
# -----------------------
def create_multi_output_target(df_in, target_col, n_days):
    df_work = df_in.copy()
    target_cols = []
    for d in range(1, n_days + 1):
        colname = f"target_day_{d}"
        df_work[colname] = df_work[target_col].shift(-d)
        target_cols.append(colname)
    df_work = df_work.dropna(subset=target_cols)
    X = df_work[feature_cols]
    y = df_work[target_cols]
    return X, y

# -----------------------
# Helper: stack estimator builder
# -----------------------
def build_stack_estimator():
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.9, random_state=42, verbosity=0)
    estimators = [("lr", lr), ("rf", rf), ("xgb", xgb)]
    meta = Ridge(alpha=1.0)
    stack = StackingRegressor(estimators=estimators, final_estimator=meta, passthrough=True, n_jobs=-1)
    return stack

# -----------------------
# 3. Train a multi-output stacking model per target (each predicts N days)
# -----------------------
models_reg = {}
metrics_summary = []

print("\nTraining multi-output stacking models for each target...")
for target in TARGETS:
    if target not in df.columns:
        print(f"Target {target} missing from dataset; skipping.")
        continue

    X, y = create_multi_output_target(df, target, N_FORECAST_DAYS)
    if len(X) < 30:
        print(f"Not enough rows to train for {target} (found {len(X)}). Skipping.")
        continue

    # time-aware split: keep order (no shuffle)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # =========================================================
    # ✨ NEW: Add StandardScaler
    # =========================================================
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    # Fit ONLY on X_train to learn the mean and standard deviation
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform X_test using the scaler fitted on X_train
    X_test_scaled = scaler.transform(X_test)
    
    # Save this scaler to use for future predictions
    scaler_filename = f"scaler_{target.replace('%','pct').replace('/','_')}.joblib"
    joblib.dump(scaler, scaler_filename)
    print(f"    Saved scaler: {scaler_filename}")
    # =========================================================

    base_stack = build_stack_estimator()
    multi_stack = MultiOutputRegressor(base_stack, n_jobs=-1)
    print(f"  Training {target} on {len(X_train)} rows...")
    
    # ✅ Train on the new SCALED training data
    multi_stack.fit(X_train_scaled, y_train)

    # ✅ Predict using the new SCALED test data
    pred_test = multi_stack.predict(X_test_scaled)
    r2_d1 = r2_score(y_test.iloc[:, 0], pred_test[:, 0])
    rmse_all = np.sqrt(mean_squared_error(y_test.values, pred_test))
    print(f"    -> {target}: Day1 R2={r2_d1:.3f}, RMSE_all={rmse_all:.3f}")

    models_reg[target] = multi_stack
    metrics_summary.append((target, r2_d1, rmse_all))

    joblib.dump(multi_stack, f"multi_output_stack_{target.replace('%','pct').replace('/','_')}.joblib")
    print(f"    Saved model: multi_output_stack_{target}.joblib")

# -----------------------
# 4. Train Rain classifier (optional)
# -----------------------
model_rain = None
if "Rainfall_mm" in df.columns:
    clf_df = df.dropna(subset=[RAIN_CLASS_COL])
    Xc = clf_df[feature_cols]
    yc = clf_df[RAIN_CLASS_COL].astype(int)
    if len(Xc) >= 40:
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, shuffle=False)
        clf = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                            use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0)
        clf.fit(Xc_train, yc_train)
        pred = clf.predict(Xc_test)
        acc = accuracy_score(yc_test, pred)
        print(f"\nRain classifier accuracy: {acc:.3f}")
        joblib.dump(clf, "model_RainToday.joblib")
        model_rain = clf
    else:
        print("\nNot enough rows to train robust Rain classifier; skipping.")

# -----------------------
# 5. Prepare latest features for forecasting
# -----------------------
print("\nPreparing latest features for forecasting...")
latest_row = df.iloc[-1:].copy()
latest_features = latest_row[feature_cols].copy()

# set future anomaly flags/severity to 0 (assume normal future)
for an in expected_anom_cols:
    if an in latest_features.columns:
        latest_features[an] = 0
for sv in expected_sev_cols:
    if sv in latest_features.columns:
        latest_features[sv] = 0.0

# ensure correct ordering/dtypes
latest_features = latest_features.astype(float)
latest_features = latest_features.reindex(columns=feature_cols, fill_value=0.0)

# -----------------------
# 6. Generate 7-day forecast using each multi-output model
# -----------------------
print(f"\nGenerating {N_FORECAST_DAYS}-day forecast...")
future_dates = [df["Date"].max() + timedelta(days=i) for i in range(1, N_FORECAST_DAYS + 1)]
forecast_rows = []

# get predicted vectors for each target
pred_vectors = {}
print(f"\n  Applying scaling to features for prediction...")
for target, model in models_reg.items():
    
    # =========================================================
    # ✨ NEW: Load and apply the correct scaler
    # =========================================================
    latest_features_scaled = latest_features[feature_cols].values # Default to unscaled
    
    try:
        scaler_filename = f"scaler_{target.replace('%','pct').replace('/','_')}.joblib"
        scaler = joblib.load(scaler_filename)
        
        # Transform the latest features using the loaded scaler
        # We use [feature_cols] to ensure column order is correct
        latest_features_scaled = scaler.transform(latest_features[feature_cols])
        
    except FileNotFoundError:
        print(f"    Warning: Scaler file {scaler_filename} not found. Predicting with unscaled data for {target}.")
    except Exception as e:
        print(f"    Error loading/using scaler {scaler_filename}: {e}. Predicting with unscaled data for {target}.")

    # ✅ Predict using the SCALED features
    pred = model.predict(latest_features_scaled)  # shape (1, N_FORECAST_DAYS)
    # =========================================================
    
    pred_vectors[target] = pred.flatten().tolist()

for i in range(N_FORECAST_DAYS):
    row = {"Date": future_dates[i]}
    for target in TARGETS:
        row[target] = float(pred_vectors.get(target, [np.nan]*N_FORECAST_DAYS)[i])

    # Rain alert & probability
    if model_rain is not None:
        # prepare classifier features (ensure same columns)
        clf_feat_cols = model_rain.get_booster().feature_names if hasattr(model_rain, "get_booster") else feature_cols
        fclf = latest_features.reindex(columns=clf_feat_cols, fill_value=0.0)
        prob = model_rain.predict_proba(fclf)[0][1]
        pred_rain = int(prob >= 0.5)
        row["Rain_Probability"] = round(float(prob) * 100.0, 1)
        row["Rain_Alert"] = "Rain Likely 🌧️" if pred_rain == 1 else "No Rain ☀️"
    else:
        rainfall = row.get("Rainfall_mm", np.nan)
        row["Rain_Probability"] = None
        row["Rain_Alert"] = "Rain Likely 🌧️" if (not pd.isna(rainfall) and rainfall > RAIN_THRESHOLD_MM) else "No Rain ☀️"

    # convert numpy types
    for k, v in list(row.items()):
        if isinstance(v, (np.floating, np.float32, np.float64)):
            row[k] = float(v)
        if isinstance(v, (np.integer, np.int32, np.int64)):
            row[k] = int(v)

    forecast_rows.append(row)

df_forecast = pd.DataFrame(forecast_rows)

# -----------------------
# 7. Save forecast JSON (local) and print preview
# -----------------------
out_path = f"forecast_{N_FORECAST_DAYS}days_with_alerts.json"
df_forecast["Date"] = pd.to_datetime(df_forecast["Date"]).dt.strftime("%Y-%m-%d")
df_forecast.to_json(out_path, orient="records", indent=4)
print(f"\nSaved forecast → {out_path}")
print("\nForecast preview:")
print(df_forecast.head())

# -----------------------
# 8. Summary metrics
# -----------------------
print("\nModel summary (regression targets):")
for t, r2, rmse in metrics_summary:
    print(f" - {t}: Day1 R2={r2:.3f}, RMSE_all={rmse:.3f}")

print("\nDone.")
    
