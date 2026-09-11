"""Baseline throughput/latency predictor: scikit-learn GradientBoostingRegressor.

    python train_sklearn_regressor.py

Trains one regressor per target (throughput_tok_s, p99_latency_ms) on the
combined vLLM + SGLang benchmark rows, reports MAE on a held-out split, and
saves both models to perf_modeling/results/ for compare_models.py.
"""
from pathlib import Path

import joblib
from data import TARGET_COLUMNS, encode_features, load_bench_data, require_enough_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

OUT_DIR = Path(__file__).resolve().parent / "results"


def main():
    df = load_bench_data()
    require_enough_data(df)
    x, y, feature_names = encode_features(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, target in enumerate(TARGET_COLUMNS):
        model = GradientBoostingRegressor(random_state=42)
        model.fit(x_train, y_train[:, i])
        preds = model.predict(x_test)
        mae = mean_absolute_error(y_test[:, i], preds)
        joblib.dump(model, OUT_DIR / f"sklearn_{target}.joblib")
        print(f"[sklearn] {target}: MAE={mae:.2f}")

    joblib.dump(feature_names, OUT_DIR / "feature_names.joblib")


if __name__ == "__main__":
    main()
