import json
import pandas as pd
import numpy as np 
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def evaluate_model(model_path: Path, test_path: Path, cat_encoders_path: Path):
    df = pd.read_csv(test_path)

    # Load category encoders (list of categories per col)
    with open(cat_encoders_path, 'r') as f:
        cat_encoders = json.load(f)

    y_test = df.iloc[:, 0]
    X_test = df.iloc[:, 1:]

    # Apply category encoding to each categorical column
    for col, cats in cat_encoders.items():
        if col in X_test.columns:
            X_test[col] = pd.Categorical(X_test[col], categories=cats)
            X_test[col] = X_test[col].cat.codes
        else:
            print(f"Warning: column '{col}' not found in test data")
            
    model = xgb.Booster()
    model.load_model(str(model_path))

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return y_test, y_pred

def plot_results(y_test, y_pred, out_path: Path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='navy')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\n Plot saved to: {out_path}")


def main():
    base = Path().resolve().parent
    model_path = base / "model" / "xgboost-model"
    test_path = base / "data" / "test.csv"
    cat_encoders_path = base / "data" / "cat_encoders.json"  
    out_path = base / "reports" / "eval_plot.png"

    y_test, y_pred = evaluate_model(model_path, test_path, cat_encoders_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(y_test, y_pred, out_path)
    
if __name__ == "__main__":
    main()

"""
Evaluation Metrics:
RMSE: 19342.6926
MAE: 12198.8901
R2 Score: -0.1576
"""