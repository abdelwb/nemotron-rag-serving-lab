#!/usr/bin/env python
"""Throughput/latency predictor: a small TensorFlow/Keras MLP.

    python train_tf_model.py

Trains one small MLP with two output heads (throughput_tok_s,
p99_latency_ms) on the combined vLLM + SGLang benchmark rows, reports MAE on
a held-out split, and saves the model to perf_modeling/results/ for
compare_models.py.
"""
from pathlib import Path

import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import TARGET_COLUMNS, encode_features, load_bench_data, require_enough_data

OUT_DIR = Path(__file__).resolve().parent / "results"


def build_model(n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(len(TARGET_COLUMNS))(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def main():
    df = load_bench_data()
    require_enough_data(df)
    x, y, feature_names = encode_features(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Small dataset + gradient descent -> standardizing inputs and targets
    # matters far more here than for the tree-based sklearn baseline.
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_s = x_scaler.transform(x_train)
    x_test_s = x_scaler.transform(x_test)
    y_train_s = y_scaler.transform(y_train)

    model = build_model(n_features=x_train_s.shape[1])
    model.fit(
        x_train_s,
        y_train_s,
        validation_split=0.2,
        epochs=200,
        batch_size=8,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
    )

    preds_s = model.predict(x_test_s, verbose=0)
    preds = y_scaler.inverse_transform(preds_s)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, target in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(y_test[:, i], preds[:, i])
        print(f"[tensorflow] {target}: MAE={mae:.2f}")

    model.save(OUT_DIR / "tf_model.keras")
    joblib.dump(x_scaler, OUT_DIR / "tf_x_scaler.joblib")
    joblib.dump(y_scaler, OUT_DIR / "tf_y_scaler.joblib")
    joblib.dump(feature_names, OUT_DIR / "feature_names.joblib")


if __name__ == "__main__":
    main()
