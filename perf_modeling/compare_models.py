"""Side-by-side comparison of the sklearn and TensorFlow throughput/latency models.

Run after both train_sklearn_regressor.py and train_tf_model.py.

    python compare_models.py
    python compare_models.py --engine vllm --batch-size 16 --input-len 512 --output-len 128
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf
from data import TARGET_COLUMNS, encode_features, load_bench_data, require_enough_data
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_models():
    sklearn_models = {t: joblib.load(RESULTS_DIR / f"sklearn_{t}.joblib") for t in TARGET_COLUMNS}
    tf_model = tf.keras.models.load_model(RESULTS_DIR / "tf_model.keras")
    x_scaler = joblib.load(RESULTS_DIR / "tf_x_scaler.joblib")
    y_scaler = joblib.load(RESULTS_DIR / "tf_y_scaler.joblib")
    feature_names = joblib.load(RESULTS_DIR / "feature_names.joblib")
    return sklearn_models, tf_model, x_scaler, y_scaler, feature_names


def evaluate_on_holdout():
    df = load_bench_data()
    require_enough_data(df)
    x, y, _ = encode_features(df)
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    sklearn_models, tf_model, x_scaler, y_scaler, _ = load_models()

    print(f"{'target':<20}{'sklearn MAE':<15}{'tensorflow MAE':<15}")
    for i, target in enumerate(TARGET_COLUMNS):
        sk_pred = sklearn_models[target].predict(x_test)
        sk_mae = mean_absolute_error(y_test[:, i], sk_pred)

        tf_pred_s = tf_model.predict(x_scaler.transform(x_test), verbose=0)
        tf_pred = y_scaler.inverse_transform(tf_pred_s)[:, i]
        tf_mae = mean_absolute_error(y_test[:, i], tf_pred)

        print(f"{target:<20}{sk_mae:<15.2f}{tf_mae:<15.2f}")


def predict_one(engine: str, batch_size: int, input_len: int, output_len: int):
    sklearn_models, tf_model, x_scaler, y_scaler, feature_names = load_models()

    row = pd.DataFrame([{
        "engine": engine, "batch_size": batch_size, "input_len": input_len, "output_len": output_len,
    }])
    encoded = pd.get_dummies(row, columns=["engine"])
    encoded = encoded.reindex(columns=feature_names, fill_value=0)
    x = encoded.to_numpy(dtype="float32")

    print(f"\nPrediction for engine={engine}, batch_size={batch_size}, input_len={input_len}, output_len={output_len}:")
    for i, target in enumerate(TARGET_COLUMNS):
        sk_pred = sklearn_models[target].predict(x)[0]
        print(f"  sklearn    {target}: {sk_pred:.1f}")

    tf_pred_s = tf_model.predict(x_scaler.transform(x), verbose=0)
    tf_pred = y_scaler.inverse_transform(tf_pred_s)[0]
    for i, target in enumerate(TARGET_COLUMNS):
        print(f"  tensorflow {target}: {tf_pred[i]:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=None, choices=["vllm", "sglang"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    args = parser.parse_args()

    if args.engine is not None:
        predict_one(args.engine, args.batch_size, args.input_len, args.output_len)
    else:
        evaluate_on_holdout()
