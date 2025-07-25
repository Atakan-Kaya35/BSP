
# -*- coding: utf-8 -*-
"""BSP LSTM Layers Model Fitter with Optuna Bayesian Optimization
Automatically generated.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import optuna
import os

def init_data(seq_len):
    CSV_FILE = "atakanka350@gmail.com.csv"
    CSV_METADATA_SKIP = 25
    TIME_COL = "Zaman damgası (GG-AA-YYYY/ss:dd:sn)"
    GLUCOSE_COL = "Glikoz Değeri (mg/dL)"

    df_raw = pd.read_csv(CSV_FILE, sep=";")
    df_raw.iloc[CSV_METADATA_SKIP:, 7] = df_raw.iloc[CSV_METADATA_SKIP:, 7].replace({"Yüksek": 400, "Düşük": 40})
    df_raw[GLUCOSE_COL] = pd.to_numeric(df_raw[GLUCOSE_COL], errors="coerce")

    df_raw[TIME_COL] = pd.to_datetime(df_raw[TIME_COL], errors='coerce')
    df_raw["hour"] = df_raw[TIME_COL].dt.hour
    df_raw["minute"] = df_raw[TIME_COL].dt.minute
    df_raw["sin_time"] = np.sin(2 * np.pi * (df_raw["hour"] * 60 + df_raw["minute"]) / 1440)
    df_raw["cos_time"] = np.cos(2 * np.pi * (df_raw["hour"] * 60 + df_raw["minute"]) / 1440)

    df_clean = df_raw.dropna(subset=[GLUCOSE_COL, "sin_time", "cos_time"]).iloc[CSV_METADATA_SKIP:]

    sc_glucose = MinMaxScaler()
    sc_time = MinMaxScaler()
    glucose_scaled = sc_glucose.fit_transform(df_clean[[GLUCOSE_COL]])
    sin_scaled = sc_time.fit_transform(df_clean[["sin_time"]])
    cos_scaled = sc_time.fit_transform(df_clean[["cos_time"]])

    data = np.hstack((glucose_scaled, sin_scaled, cos_scaled))

    def create_sequences(data_array, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data_array)):
            X.append(data_array[i-seq_length:i])
            y.append(data_array[i, 0])
        return np.array(X), np.array(y)

    X_sequences_all, y_sequences_all = create_sequences(data, seq_len)
    return X_sequences_all, y_sequences_all, sc_glucose


def build_and_train_model(trial):
    LSTM_units = trial.suggest_int('LSTM_units', 16, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.0001, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    num_layers = trial.suggest_int('num_layers', 0, 30)
    batch_size = trial.suggest_categorical('batch_size', [16, 24, 32])
    epochs = trial.suggest_int('epochs', 50, 120, step=10)
    seq_len = trial.suggest_int('seq_len', 6, 24)
    
    X_sequences_all, y_sequences_all, sc_glucose = init_data(seq_len)
    TEST_SPLIT_RATIO = 0.1
    VAL_SPLIT_RATIO = 0.1
    test_size = int(len(X_sequences_all) * TEST_SPLIT_RATIO)
    X_train_val = X_sequences_all[:-test_size]
    y_train_val = y_sequences_all[:-test_size]
    val_size = int(len(X_train_val) * VAL_SPLIT_RATIO)
    X_train = X_train_val[:-val_size]
    y_train = y_train_val[:-val_size]
    X_val = X_train_val[-val_size:]
    y_val = y_train_val[-val_size:]

    print("Starting training for:", trial)
    model = Sequential()
    model.add(LSTM(units=LSTM_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout))
    for _ in range(num_layers):
        model.add(LSTM(units=LSTM_units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(units=LSTM_units))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    return min(history.history["val_loss"])


def objective(trial):

    return build_and_train_model(trial)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # 50 trials for now

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (val_loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save results to CSV
    df = study.trials_dataframe()
    df.to_csv("optuna_hyperparam_results.csv", index=False)
