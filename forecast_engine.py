"""
forecast_engine.py

Modular forecasting pipeline for LNG prices using Conv1D + LSTM.

Supports:
- Arbitrary time series (e.g. Henry Hub, TTF, JKM...)
- Arbitrary forecast horizons (e.g. 30 / 60 / 90 days ahead)
- Train/validation split by date
- Horizon-aware windowing
- Model training with EarlyStopping
- Validation metrics: MAE, RMSE, MAPE, R²
- Optional retraining on full history for production use
- One-step horizon forecast using the latest window (e.g. t + 30 days)

Author: (You 😊)
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import joblib


# ---------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1000
LEARNING_RATE = 1e-4
EPOCHS_SELECTION = 50        # for train/validation stage
EPOCHS_FULL = 50             # for retrain on full history
PATIENCE = 5                 # early stopping patience

# Output directories (will be created if missing)
MODELS_DIR = Path("models")
SCALERS_DIR = Path("scalers")
RESULTS_DIR = Path("results")
METRICS_DIR = Path("metrics")

for d in [MODELS_DIR, SCALERS_DIR, RESULTS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utility: windowed dataset with horizon
# ---------------------------------------------------------------------
def make_lng_forecast_dataset(series, window_size, horizon,
                              batch_size=BATCH_SIZE,
                              shuffle_buffer=SHUFFLE_BUFFER,
                              shuffle=True):
    """
    Create a tf.data.Dataset for LNG price forecasting.

    Args:
        series (1D np.array): scaled time series
        window_size (int): number of past steps used as input
        horizon (int): forecast horizon (predict value at t + horizon)
        batch_size (int)
        shuffle_buffer (int)
        shuffle (bool): whether to shuffle the dataset

    Returns:
        tf.data.Dataset yielding (X, y) pairs with:
            X.shape = (batch, window_size, 1)
            y.shape = (batch,)
    """
    series = np.asarray(series).astype(np.float32)
    total_window = window_size + horizon

    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(total_window, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(total_window))

    # X = first window_size steps, y = last element (at t+window_size+horizon-1)
    ds = ds.map(lambda w: (tf.expand_dims(w[:window_size], axis=-1), w[-1]))

    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# Utility: build model
# ---------------------------------------------------------------------
def build_model(window_size, horizon, learning_rate=LEARNING_RATE):
    """
    Build a Conv1D + LSTM + Dense model for horizon forecasting.

    Args:
        window_size (int)
        horizon (int)  # not used directly in architecture, but kept for clarity
        learning_rate (float)

    Returns:
        compiled tf.keras.Model
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu"),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    return model


# ---------------------------------------------------------------------
# Utility: validation evaluation (val_true, val_pred, val_dates)
# ---------------------------------------------------------------------
def evaluate_on_validation(model,
                           train_df,
                           valid_df,
                           scaler_train,
                           window_size,
                           horizon,
                           price_col="Price",
                           date_col="Date"):
    """
    Compute validation predictions and metrics using a trained model.

    Returns:
        val_dates (pd.DatetimeIndex)
        val_true (np.array)
        val_pred (np.array)
        metrics dict {mae, rmse, mape, R2}
    """
    # Scale validation prices using train-fitted scaler
    valid_scaled = scaler_train.transform(valid_df[[price_col]]).astype(np.float32)
    valid_series = valid_scaled.flatten()

    total_window = window_size + horizon
    num_windows = len(valid_series) - total_window + 1

    X_list, y_list = [], []
    for i in range(num_windows):
        window = valid_series[i:i + total_window]
        X_list.append(window[:window_size])
        y_list.append(window[-1])

    val_x = np.array(X_list, dtype=np.float32)[..., np.newaxis]
    val_y = np.array(y_list, dtype=np.float32)

    # Predict in scaled space
    y_pred_scaled = model.predict(val_x, verbose=0).flatten()

    # Inverse transform
    # Important: scaler_train was fit on train_df[price_col]
    val_true = scaler_train.inverse_transform(val_y.reshape(-1, 1)).flatten()
    val_pred = scaler_train.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Validation dates
    valid_dates_all = valid_df[date_col].reset_index(drop=True)
    start_idx = window_size + horizon - 1
    val_dates = valid_dates_all[start_idx:start_idx + len(val_true)]

    # Metrics
    mae = mean_absolute_error(val_true, val_pred)
    rmse = np.sqrt(mean_squared_error(val_true, val_pred))
    mape = np.mean(np.abs((val_true - val_pred) / val_true)) * 100
    R2 = r2_score(val_true, val_pred)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "R2": float(R2)
    }

    return val_dates, val_true, val_pred, metrics


# ---------------------------------------------------------------------
# Utility: one-step horizon forecast using full history model
# ---------------------------------------------------------------------
def forecast_future_one_step(model,
                             scaler_full,
                             df,
                             window_size,
                             horizon,
                             price_col="Price",
                             date_col="Date"):
    """
    Use the final model trained on full history to produce a single
    horizon forecast:

        predict price at (last_date + horizon days)
        using the most recent `window_size` days.

    Returns:
        future_date (pd.Timestamp)
        future_value (float)
    """
    series_scaled = scaler_full.transform(df[[price_col]]).astype(np.float32).flatten()

    if len(series_scaled) < window_size:
        raise ValueError("Not enough data points to form a full window.")

    window = series_scaled[-window_size:]
    x_input = window.reshape(1, window_size, 1)

    pred_scaled = model.predict(x_input, verbose=0)[0, 0]
    future_value = scaler_full.inverse_transform([[pred_scaled]])[0, 0]

    last_date = df[date_col].iloc[-1]
    future_date = last_date + pd.Timedelta(days=horizon)

    return future_date, float(future_value)


# ---------------------------------------------------------------------
# Main pipeline: run_forecast
# ---------------------------------------------------------------------
def run_forecast(
    name,
    df,
    window_size,
    horizon,
    cutoff_date,
    price_col="Price",
    date_col="Date",
    retrain_on_full=True,
):
    """
    Full pipeline for a single (market, horizon) combination.

    Steps:
    1) Sort by date, split train/validation.
    2) Fit scaler on train, build train/validation datasets.
    3) Train model for selection (train vs validation).
    4) Evaluate on validation (val_true, val_pred, val_dates, metrics).
    5) Optionally retrain best architecture on full history.
    6) Use full-history model to forecast one horizon ahead.
    7) Save model, scaler, metrics, and results CSV.

    Args:
        name (str): e.g. "HH", "TTF", "JKM"
        df (pd.DataFrame): must contain [date_col, price_col]
        window_size (int)
        horizon (int)
        cutoff_date (str or Timestamp): train/validation split date
        price_col (str)
        date_col (str)
        retrain_on_full (bool)

    Returns:
        metrics (dict), future_date (Timestamp), future_value (float)
    """
    print(f"\n==== Running forecast for {name}, horizon={horizon} ====")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # 1) Split train and validation
    train_df = df[df[date_col] <= cutoff_date].copy()
    valid_df = df[df[date_col] > cutoff_date].copy()

    if len(train_df) < window_size + horizon:
        raise ValueError("Training data too short for given window_size and horizon.")

    if len(valid_df) < window_size + horizon:
        print("Warning: validation period is quite short for this window & horizon.")

    # 2) Fit scaler on train_only for selection stage
    scaler_train = MinMaxScaler()
    train_scaled = scaler_train.fit_transform(train_df[[price_col]]).astype(np.float32)
    train_series = train_scaled.flatten()

    valid_scaled = scaler_train.transform(valid_df[[price_col]]).astype(np.float32)
    valid_series = valid_scaled.flatten()

    # 3) Build datasets
    train_ds = make_lng_forecast_dataset(train_series, window_size, horizon,
                                         batch_size=BATCH_SIZE,
                                         shuffle_buffer=SHUFFLE_BUFFER,
                                         shuffle=True)

    val_ds = make_lng_forecast_dataset(valid_series, window_size, horizon,
                                       batch_size=BATCH_SIZE,
                                       shuffle_buffer=SHUFFLE_BUFFER,
                                       shuffle=False)

    # 4) Model for selection (train/validation)
    model_sel = build_model(window_size, horizon, learning_rate=LEARNING_RATE)

    ckpt_path = MODELS_DIR / f"{name}_h{horizon}_selection.keras"
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    cp = ModelCheckpoint(filepath=str(ckpt_path),
                         monitor="val_loss",
                         save_best_only=True,
                         save_weights_only=False)

    print("Training selection model (train vs validation)...")
    model_sel.fit(
        train_ds,
        epochs=EPOCHS_SELECTION,
        validation_data=val_ds,
        callbacks=[es, cp],
        verbose=1
    )

    # 5) Evaluate on validation
    print("Evaluating on validation...")
    val_dates, val_true, val_pred, metrics = evaluate_on_validation(
        model_sel, train_df, valid_df, scaler_train,
        window_size, horizon,
        price_col=price_col,
        date_col=date_col
    )

    print(
        f"Validation metrics [{name}, h={horizon}]: "
        f"MAE={metrics['mae']:.3f}, "
        f"RMSE={metrics['rmse']:.3f}, "
        f"MAPE={metrics['mape']:.2f}%, "
        f"R2={metrics['R2']:.3f}"
    )

    # 6) Optionally retrain on full history for production
    if retrain_on_full:
        print("Retraining best architecture on FULL history for production use...")
        scaler_full = MinMaxScaler()
        full_scaled = scaler_full.fit_transform(df[[price_col]]).astype(np.float32)
        full_series = full_scaled.flatten()

        full_ds = make_lng_forecast_dataset(full_series, window_size, horizon,
                                            batch_size=BATCH_SIZE,
                                            shuffle_buffer=SHUFFLE_BUFFER,
                                            shuffle=True)

        model_full = build_model(window_size, horizon, learning_rate=LEARNING_RATE)
        es_full = EarlyStopping(monitor="loss", patience=PATIENCE, restore_best_weights=True)

        model_full.fit(
            full_ds,
            epochs=EPOCHS_FULL,
            callbacks=[es_full],
            verbose=1
        )

        # Save full-history model & scaler
        model_path = MODELS_DIR / f"{name}_h{horizon}_full.keras"
        scaler_path = SCALERS_DIR / f"{name}_h{horizon}_scaler.pkl"

        model_full.save(model_path)
        joblib.dump(scaler_full, scaler_path)

        print(f"Saved full model to {model_path}")
        print(f"Saved scaler to {scaler_path}")

        # 7) Future one-step horizon forecast
        future_date, future_value = forecast_future_one_step(
            model_full, scaler_full, df,
            window_size, horizon,
            price_col=price_col,
            date_col=date_col
        )

    else:
        # Use selection model & train-scaler for future forecast (less ideal)
        model_full = model_sel
        scaler_full = scaler_train
        future_date, future_value = forecast_future_one_step(
            model_full, scaler_full, df,
            window_size, horizon,
            price_col=price_col,
            date_col=date_col
        )

    print(f"Future forecast [{name}, h={horizon}]: "
          f"{future_date.date()} → {future_value:.3f}")

    # 8) Save results (validation + future) to CSV
    results_rows = []

    # Validation rows
    for d, y_true, y_hat in zip(val_dates, val_true, val_pred):
        results_rows.append({
            "market": name,
            "horizon": horizon,
            "segment": "validation",
            "date": d,
            "actual": y_true,
            "predicted": y_hat
        })

    # Future forecast row
    results_rows.append({
        "market": name,
        "horizon": horizon,
        "segment": "future",
        "date": future_date,
        "actual": None,
        "predicted": future_value
    })

    results_df = pd.DataFrame(results_rows)
    results_path = RESULTS_DIR / f"{name}_h{horizon}_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    # 9) Save metrics JSON
    metrics_path = METRICS_DIR / f"{name}_h{horizon}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return metrics, future_date, future_value


# ---------------------------------------------------------------------
# Example driver (you can adapt this to your real LNG files)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """
    Example usage:
    Assume you have CSVs like:
        data/HH_Price.csv  with columns ["Date", "HH_Price"]
        data/TTF_Price.csv with columns ["Date", "TTF_Price"]
    You can adapt below mappings to your actual filenames / columns.
    """

    DATA_DIR = Path("data")

    # Example configuration: adapt to your filenames & columns
    configs = [
        {
            "name": "TTF",
            "csv": DATA_DIR / "TTF_price.csv",
            "price_col": "TTF_Price",
            "date_col": "Date",
        },
        {
            "name": "JKM",
            "csv": DATA_DIR / "JKM_price.csv",
            "price_col": "JKM_Price",
            "date_col": "Date",
        },
        {
            "name": "PVB",
            "csv": DATA_DIR / "PVB_price.csv",
            "price_col": "PVB_Price",
            "date_col": "Date",
        },
        # Add other LNG benchmarks here:
        # {
        #     "name": "TTF",
        #     "csv": DATA_DIR / "TTF_Price.csv",
        #     "price_col": "TTF_Price",
        #     "date_col": "Date",
        # },
        # ...
    ]

    horizons = [30, 60, 90]
    window_size = 90
    cutoff_date = "2021-12-31"

    for cfg in configs:
        df_market = pd.read_csv(cfg["csv"], parse_dates=[cfg["date_col"]])
        for h in horizons:
            metrics, fdate, fval = run_forecast(
                name=cfg["name"],
                df=df_market,
                window_size=window_size,
                horizon=h,
                cutoff_date=cutoff_date,
                price_col=cfg["price_col"],
                date_col=cfg["date_col"],
                retrain_on_full=True
            )
