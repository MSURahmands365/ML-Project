
#!/usr/bin/env python3
"""
Runs PakWheels Pipeline:
Load → Preprocess → Train/Test Split → Fit Encoders (Train-only) → Transform → Tune (optional) → Train → Evaluate
Also supports: Predict on new (UI) data using saved encoders & feature order.

Usage:
  # Full pipeline
  python scripts/run_pipeline.py pipeline --input data/raw/PakWheelsDataSet.csv --target Price --tune

  # Predict using saved encoders + feature order
  python scripts/run_pipeline.py predict --input data/new_inputs.csv --target Price --model_uri <mlflow_model_uri>

Notes:
- Encoders are fit on TRAIN ONLY to avoid leakage.
- Feature column order is saved to artifacts/feature_columns.json for consistent UI inference.
"""

import os
import sys
import json
import time
import argparse
import joblib
import pandas as pd

from pathlib import Path
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# === Fix import path for local modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules
from src.data.preprocess import preprocess_data
from src.features.build_features import fit_and_save_encoders  # We fit on train only here
from src.models.train import train_model
from src.models.tune import tune_model
from src.models.evaluate import evaluate_model


# ---------- Utilities ----------
def project_paths():
    """Return canonical project paths."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return {
        "project_root": base,
        "artifacts_dir": os.path.join(base, "artifacts"),
        "processed_dir": os.path.join(base, "data", "processed"),
        "target_encoder": os.path.join(base, "artifacts", "target_encoder.pkl"),
        "one_hot_encoder": os.path.join(base, "artifacts", "one_hot_encoder.pkl"),
        "feature_cols_json": os.path.join(base, "artifacts", "feature_columns.json"),
    }



def setup_mlflow(args):
    # Use SQLite DB file in project root by default
    paths = project_paths()
    tracking_uri = args.mlflow_uri or f"sqlite:///{Path(paths['project_root'])/'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)



def transform_with_fitted_encoders(df: pd.DataFrame, target_enc, ohe_enc) -> pd.DataFrame:
    """
    Mirror the logic inside fit_and_save_encoders:
    - Target encode ['Name', 'City'] together (encoder was fitted with BOTH columns).
    - One-hot encode ['Make', 'Transmission', 'Engine Type'] together (encoder fitted with ALL THREE).
    - Drop original OHE columns and merge new binary columns.
    """
    df = df.copy()

    # --- Target Encoding (fit was on ['Name', 'City']) ---
    te_cols = ['Name', 'City']
    # Ensure both columns exist (add missing with None/NaN so shape matches)
    for col in te_cols:
        if col not in df.columns:
            df[col] = None
    # Transform BOTH columns together to satisfy encoder's expected dimension
    df[te_cols] = target_enc.transform(df[te_cols])

    # --- One-Hot Encoding (fit was on ['Make', 'Transmission', 'Engine Type']) ---
    ohe_cols = ['Make', 'Transmission', 'Engine Type']
    for col in ohe_cols:
        if col not in df.columns:
            df[col] = None
    df_ohe_only = ohe_enc.transform(df[ohe_cols])

    # Drop original text columns and join the new OHE columns
    df.drop(columns=ohe_cols, inplace=True)
    # Safer concat preserves index alignment
    df = pd.concat([df, df_ohe_only], axis=1)

    return df



def ensure_feature_order(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Ensure the df has exactly the columns in feature_cols in the same order.
    Missing columns are added with 0; extra columns are dropped.
    """
    df = df.copy()
    # Add missing
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    # Drop extras
    extras = [c for c in df.columns if c not in feature_cols]
    if extras:
        df.drop(columns=extras, inplace=True)
    # Reorder
    df = df[feature_cols]
    return df


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------- Pipeline ----------
def run_pipeline(args):
    setup_mlflow(args)
    paths = project_paths()

    os.makedirs(paths["artifacts_dir"], exist_ok=True)
    os.makedirs(paths["processed_dir"], exist_ok=True)

    with mlflow.start_run():
        mlflow.log_param("model_type", "xgboost_regressor")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("min_price", args.min_price)
        mlflow.log_param("tune", bool(args.tune))

        # 1) Load
        print("🔄 Loading raw data...")
        df = pd.read_csv(args.input)
        print(f"✅ Raw loaded: {df.shape[0]} rows")

        # 2) Preprocess
        print("🔧 Preprocessing...")
        df = preprocess_data(df, target_col=args.target)

        # Manual filters (aligned with your prepare script)
        df = df[df[args.target] > args.min_price].drop_duplicates()
        processed_path = os.path.join(paths["processed_dir"], "pakwheels_processed.csv")
        df.to_csv(processed_path, index=False)
        print(f"✅ Preprocessed saved → {processed_path}")

        # 3) Split BEFORE encoding (to avoid leakage)
        print("📊 Splitting data (train/test)...")
        X = df.drop(columns=[args.target])
        y = df[args.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )

        # 4) Fit encoders on TRAIN ONLY, using your build_features module
        print("🛠️ Fitting encoders on TRAIN ONLY...")
        df_train_for_fit = pd.concat([X_train, y_train], axis=1)  # includes target for target-encoding
        df_train_encoded = fit_and_save_encoders(df_train_for_fit, target_col=args.target)

        # Move saved encoders to artifacts/ (if they were saved in CWD)
        # (We expect fit_and_save_encoders to have created 'target_encoder.pkl' and 'one_hot_encoder.pkl' in CWD)
        if os.path.exists("target_encoder.pkl"):
            os.replace("target_encoder.pkl", paths["target_encoder"])
        if os.path.exists("one_hot_encoder.pkl"):
            os.replace("one_hot_encoder.pkl", paths["one_hot_encoder"])

        # Load encoders
        target_enc = joblib.load(paths["target_encoder"])
        ohe_enc = joblib.load(paths["one_hot_encoder"])

        # 5) Transform TRAIN and TEST with fitted encoders
        print("🔄 Transforming TRAIN/TEST with fitted encoders...")
        # TRAIN encoded already returned but includes target. Recreate features from df_train_encoded.
        feature_cols = list(df_train_encoded.drop(columns=[args.target]).columns)

        X_train_encoded = df_train_encoded.drop(columns=[args.target]).copy()
        # TEST encoding
        df_test_for_transform = X_test.copy()
        X_test_transformed = transform_with_fitted_encoders(df_test_for_transform, target_enc, ohe_enc)

        # Ensure consistent feature order for TEST
        X_test_encoded = ensure_feature_order(X_test_transformed, feature_cols)

        # Convert any bool to int for XGBoost compatibility
        for c in X_train_encoded.select_dtypes(include=["bool"]).columns:
            X_train_encoded[c] = X_train_encoded[c].astype(int)
        for c in X_test_encoded.select_dtypes(include=["bool"]).columns:
            X_test_encoded[c] = X_test_encoded[c].astype(int)

        # Save & log feature order for UI predictions
        save_json(feature_cols, paths["feature_cols_json"])
        mlflow.log_artifact(paths["feature_cols_json"])

        # 6) Tuning (optional)
        final_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 7,
            "n_jobs": -1,
            "random_state": 42
        }
        if args.tune:
            print("🎛️ Tuning hyperparameters with Optuna (on TRAIN)...")
            # IMPORTANT: tune_model currently uses XGBRFRegressor; consider refactoring it.
            best_params = tune_model(X_train_encoded, y_train)
            # Normalize keys for XGBRegressor (ensure name is 'n_estimators' etc.)
            if "n_estimator" in best_params:
                best_params["n_estimators"] = best_params.pop("n_estimator")
            final_params.update(best_params)
            print("✅ Best Params:", final_params)

        # 7) Train using your train_model (logs to MLflow)
        print("🤖 Training model via src.models.train.train_model...")
        # We pass TRAIN ONLY (to keep leakage-free). train_model will do its own internal split for validation.
        df_train_encoded_full = pd.concat([X_train_encoded, y_train], axis=1)
        model = train_model(df_train_encoded_full, target_col=args.target, params=final_params)

        # 8) Evaluate on our held-out TEST using evaluate.py
        print("📊 Evaluating on held-out TEST...")
        # Predictions
        t1 = time.time()
        preds = model.predict(X_test_encoded)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        # Log metrics
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        try:
           rmse = mean_squared_error(y_test, preds, squared=False)
        
        except TypeError:
           from math import sqrt
           rmse = sqrt(mean_squared_error(y_test, preds))

        mlflow.log_metric("mae_holdout", mae)
        mlflow.log_metric("r2_holdout", r2)
        mlflow.log_metric("rmse_holdout", rmse)

        # Pretty print via your evaluate module
        evaluate_model(model, X_test_encoded, y_test)

        # 9) Save model to MLflow (already done inside train_model). Re-log encoders as artifacts.
        print("💾 Logging encoders to MLflow artifacts...")
        mlflow.log_artifact(paths["target_encoder"])
        mlflow.log_artifact(paths["one_hot_encoder"])
        mlflow.log_artifact(processed_path)

        print("✅ Pipeline complete.")


# ---------- Predict (for future UI) ----------

def run_predict(args):
    """
    Predict on new incoming data (from UI or CSV/JSON):
    - Load encoders and feature order saved during training.
    - Preprocess incoming data.
    - Transform with fitted encoders.
    - Reorder/align features.
    - Load model from MLflow URI.
    - Output predictions.
    """
    paths = project_paths()

    print("🔄 Loading new input...")
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        df_new = pd.read_csv(args.input)
    elif ext == ".json":
        df_new = pd.read_json(args.input)
    else:
        raise ValueError(f"Unsupported input format: {ext}. Use CSV or JSON.")

    # Optional: ensure required raw columns exist (fill missing with None)
    required_cols = ['Make', 'Name', 'Transmission', 'Engine Type', 'Engine Capacity(CC)', 'City', 'Year']
    missing = [c for c in required_cols if c not in df_new.columns]
    for c in missing:
        df_new[c] = None
    if missing:
        print(f"⚠️ Missing raw columns auto-filled with None: {missing}")

    print("🔧 Preprocessing incoming data...")
    df_new = preprocess_data(df_new, target_col=args.target if args.target in df_new.columns else args.target)

    print("🛠️ Loading encoders + feature order...")
    target_enc = joblib.load(paths["target_encoder"])
    ohe_enc = joblib.load(paths["one_hot_encoder"])
    with open(paths["feature_cols_json"], "r") as f:
        feature_cols = json.load(f)

    print("🔄 Transforming incoming data with fitted encoders...")
    df_new_transformed = transform_with_fitted_encoders(df_new.copy(), target_enc, ohe_enc)
    X_new = ensure_feature_order(df_new_transformed, feature_cols)

    # Convert any bools to int (XGBoost compatibility)
    for c in X_new.select_dtypes(include=["bool"]).columns:
        X_new[c] = X_new[c].astype(int)

    print("🤖 Loading model...")
    if not args.model_uri:
        raise ValueError("Please provide --model_uri (e.g., runs:/<run_id>/model).")
    model = mlflow.xgboost.load_model(args.model_uri)

    print("🎯 Predicting...")
    import xgboost as xgb

    # Ensure numeric types for Booster + DMatrix
    X_new = X_new.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Optional: strict non-numeric check
    non_numeric = X_new.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features found after coercion: {non_numeric}")

    dtest = xgb.DMatrix(X_new)
    preds = model.predict(dtest)

    out = pd.DataFrame({"prediction": preds})
    print(out.head(10).to_string(index=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"✅ Predictions saved → {args.output}")



# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="PakWheels Price Pipeline + Predict")
    subparsers = p.add_subparsers(dest="command", required=True)

    # Pipeline subcommand
    p_pipe = subparsers.add_parser("pipeline", help="Run full training pipeline")
    p_pipe.add_argument("--input", type=str, required=True, help="Path to raw CSV")
    p_pipe.add_argument("--target", type=str, default="Price", help="Target column name")
    p_pipe.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    p_pipe.add_argument("--min_price", type=float, default=100000, help="Min price filter")
    p_pipe.add_argument("--experiment", type=str, default="PakWheels Price Prediction", help="MLflow experiment name")
    p_pipe.add_argument("--mlflow_uri", type=str, default=None, help="MLflow tracking URI (file://...)")
    p_pipe.add_argument("--tune", action="store_true", help="Run Optuna tuning on training set")

    # Predict subcommand
    p_pred = subparsers.add_parser("predict", help="Predict on new (UI) data")
    p_pred.add_argument("--input", type=str, required=True, help="Path to new input CSV/JSON")
    p_pred.add_argument("--target", type=str, default="Price", help="Target column name (kept for schema compatibility)")
    p_pred.add_argument("--model_uri", type=str, required=True, help="MLflow model URI (e.g., runs:/<run_id>/model)")
    p_pred.add_argument("--output", type=str, default=None, help="Optional path to save predictions CSV")

    args = p.parse_args()

    if args.command == "pipeline":
        run_pipeline(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
