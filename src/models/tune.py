import optuna
from xgboost import XGBRFRegressor
from sklearn.model_selection import cross_val_score
def tune_model(x,y):
  def objective(trial):
    params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }
    model=XGBRFRegressor(**params)
    score=cross_val_score(model,x,y,cv=3,scoring="neg_mean_absolute_error")
    return score.mean()
  study=optuna.create_study(direction="maximize")
  study.optimize(objective,n_trials=5)
  print("best Params: ",study.best_params)
  return study.best_params
