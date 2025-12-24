
# src/models/train.py

import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(df: pd.DataFrame, target_col: str, params: dict = None):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    if params is None:
        params = {
            "n_estimators": 300,         # ✅ plural
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",  # ✅ explicit objective
        }
    else:
        # Ensure objective exists for consistency
        params.setdefault("objective", "reg:squarederror")

    model = XGBRegressor(**params)
    model.fit(x_train, y_train)  # ✅ removed stray comma

    preds = model.predict(x_val)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    # Log params/metrics into the active run (controlled by run_pipeline.py)
    mlflow.log_params(params)
    mlflow.log_metric("mae_train_split", mae)
    mlflow.log_metric("r2_train_split", r2)

    # ✅ Log the raw Booster using MLflow's xgboost flavor (avoids _estimator_type issue)
    booster = model.get_booster()
    mlflow.xgboost.log_model(booster, name="model")

    # Optional: snapshot for reproducibility
    try:
        df.to_csv("artifacts/processed_train_df.csv", index=False)
        mlflow.log_artifact("artifacts/processed_train_df.csv")
    except Exception:
        pass

    print(f" Model trained. (Train-split) MAE: {mae:,.0f}, R2: {r2:.4f}")
    return model
