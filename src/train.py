import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import optuna
import random
import mlflow
import mlflow.sklearn

random.seed(42)
np.random.seed(42)

def train(file_path):
    df = pd.read_parquet(file_path)
    common_drgs = df['DRG_Cd'].value_counts()[df['DRG_Cd'].value_counts() > 20].index
    df['drg_grouped'] = df['DRG_Cd'].apply(lambda x: x if x in common_drgs else 'Other')

    X = df.drop(columns=[
        'Tot_Dschrgs', 'Avg_Submtd_Cvrd_Chrg', 'Avg_Tot_Pymt_Amt',
        'Avg_Mdcr_Pymt_Amt', 'Rndrng_Prvdr_Org_Name', 'Rndrng_Prvdr_City',
        'Rndrng_Prvdr_St', 'Rndrng_Prvdr_State_Abrvtn'
    ])

    y = df['Avg_Mdcr_Pymt_Amt']

    categorical_cols = [
        'Rndrng_Prvdr_CCN', 'DRG_Cd',
        'RUCA_category', 'Rndrng_Prvdr_State_FIPS',
        'Rndrng_Prvdr_Zip5', 'drg_grouped', 'year'
    ]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

    y_binned = pd.qcut(y, q=5, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binned
    )

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0)
        }

        model = Pipeline([
            ("preprocess", preprocessor),
            ("xgb", XGBRegressor(**param, random_state=42))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred)

    optuna_study = optuna.create_study(direction='maximize')
    optuna_study.optimize(objective, n_trials=20)

    print("Best parameters:", optuna_study.best_params)
    print("Best R2 score:", optuna_study.best_value)

    final_model = Pipeline([
        ("preprocess", preprocessor),
        ("xgb", XGBRegressor(**optuna_study.best_params, random_state=42))
    ])

    final_model.fit(X_train, y_train)
    y_pred_optuna = final_model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred_optuna))
    print("MSE:", mean_squared_error(y_test, y_pred_optuna))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_optuna)))
    print("R²:", r2_score(y_test, y_pred_optuna))

    base_path = Path(file_path).resolve().parent.parent
    model_dir = base_path / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = joblib.dump(final_model, model_dir / "xgb_pipeline.joblib")
    print(f"Model saved at {model_path[0]}")

    with mlflow.start_run(run_name="Optuna-XGBoost"):
        mlflow.log_params(optuna_study.best_params)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_optuna))
        mae = mean_absolute_error(y_test, y_pred_optuna)
        r2 = r2_score(y_test, y_pred_optuna)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(final_model, artifact_path="model")

        print(f"MLflow run logged at: {mlflow.get_artifact_uri()}")

    return model_path[0]

def main():
    base = Path(__file__).resolve().parent.parent
    file_path = base / "data" / "cms_cleaned.parquet"
    train(file_path)


if __name__ == "__main__":
    main()
