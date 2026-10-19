"""
Model Training and Evaluation Script.
Trains and compares four regression models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regression (SVR)

Evaluates on MAE, RMSE, and R2 score, displays a comparison table,
selects the best model, and serializes artifacts to the models/ directory.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import joblib

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.preprocessing import prepare_training_data


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(y_true, y_pred)
    except ImportError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def train_and_evaluate_models(
    csv_path: Path = None,
    models_dir: Path = None,
    random_state: int = 42,
):
    csv_path = csv_path or (CURRENT_DIR / "student_performance.csv")
    models_dir = models_dir or (CURRENT_DIR / "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run generate_dataset.py first!"
        )

    print(f"Loading student dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Preprocessing and train-test split
    X_train_scaled, X_test_scaled, y_train, y_test, preprocessor = prepare_training_data(
        df=df,
        target_column="final_score",
        test_size=0.2,
        random_state=random_state,
    )

    print(f"Training set size: {X_train_scaled.shape[0]} samples")
    print(f"Testing set size:  {X_test_scaled.shape[0]} samples")
    print(f"Number of features: {X_train_scaled.shape[1]}")

    from sklearn.ensemble import RandomForestRegressor

    # Initialize 4 candidate regression models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=8,
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_split=6,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Regression (SVR)": SVR(
            kernel="rbf",
            C=15.0,
            epsilon=0.2,
        ),
    }

    results = {}
    fitted_models = {}

    print("\n" + "=" * 65)
    print("                TRAINING & EVALUATION BENCHMARK")
    print("=" * 65)

    for name, model in candidate_models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = compute_metrics(y_test, y_pred)
        results[name] = metrics
        fitted_models[name] = model

        # Save individual model checkpoint
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, models_dir / f"{safe_name}.pkl")

    # Display comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[["MAE", "RMSE", "R2"]]
    print("\n" + "=" * 65)
    print("                   MODEL COMPARISON TABLE")
    print("=" * 65)
    print(comparison_df.to_string())
    print("=" * 65)

    # Select best model: highest R2, broken by lowest RMSE / MAE
    best_model_name = max(
        results.keys(),
        key=lambda m: (results[m]["R2"], -results[m]["RMSE"], -results[m]["MAE"]),
    )
    best_model = fitted_models[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n[BEST] Best Selected Model: {best_model_name}")
    print(f"   R2 Score : {best_metrics['R2']:.4f}")
    print(f"   MAE      : {best_metrics['MAE']:.4f}")
    print(f"   RMSE     : {best_metrics['RMSE']:.4f}")

    # Serialize best model and preprocessor to required paths
    best_model_path = models_dir / "best_model.pkl"
    preprocessor_path = models_dir / "preprocessor.pkl"
    metrics_summary_path = models_dir / "model_metrics.json"

    joblib.dump(best_model, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)

    # Save metrics JSON for Streamlit dashboard and reports
    metrics_payload = {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "comparison": results,
        "feature_columns": preprocessor.feature_columns,
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
    }
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"\nSaved best model to:      {best_model_path}")
    print(f"Saved preprocessor to:    {preprocessor_path}")
    print(f"Saved metrics summary to: {metrics_summary_path}")

    return results, best_model_name


if __name__ == "__main__":
    train_and_evaluate_models()
