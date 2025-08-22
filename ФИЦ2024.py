# ============================================================
# УСТАНОВКА БИБЛИОТЕК
# ============================================================
# !pip install pandas numpy matplotlib neuralforecast scikit-learn

# ============================================================
# ИМПОРТЫ
# ============================================================
import os
from math import sqrt
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from neuralforecast.core import NeuralForecast
from neuralforecast.models import MLP, NBEATS, RNN, TSMixer, NHITS, NBEATSx, TimesNet
from neuralforecast.losses.pytorch import RMSE

# ============================================================
# КОНФИГ
# ============================================================
CSV_FILES = {
    "1 цех": "data1.csv",
    "2 цех": "data2.csv",
    "3 цех": "data3.csv",
    "4 цех": "data4.csv",
    "5 цех": "data5.csv",
}
DATE_COL = "date"
TARGET_COL = "value"
TIME_COL = "ds"
FREQ = "D"

# Параметры моделей/валидации
H = 30
INPUT_SIZE = 60
VAL_SIZE = 30
MAX_STEPS = 100

# ============================================================
# УТИЛИТЫ
# ============================================================
def load_series(csv_path: str, uid: str) -> pd.DataFrame:
    """Читает CSV, делает колонки (ds, value, unique_id)."""
    df = pd.read_csv(csv_path)
    if DATE_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"{csv_path}: ожидались колонки '{DATE_COL}' и '{TARGET_COL}'")
    df = df.rename(columns={DATE_COL: TIME_COL})
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df[[TIME_COL, TARGET_COL]].copy()
    df["unique_id"] = uid
    return df

def fill_na_rolling_mean(df: pd.DataFrame, col: str, window: int = 2) -> pd.DataFrame:
    """Заполняет NaN скользящим средним по соседям (min_periods=1)."""
    s = df[col]
    df[col] = s.fillna(s.rolling(window=window, min_periods=1).mean())
    return df

def plot_forecast(test_df: pd.DataFrame, pred_df: pd.DataFrame, title: str = ""):
    """
    Отрисовка фактов и всех модельных прогнозов.
    test_df: DataFrame с колонками ['ds','value']
    pred_df: DataFrame c колонками ['ds', 'unique_id', <имена моделей>...]
    """
    plt.figure(figsize=(16, 6))
    # Факты
    plt.plot(test_df["ds"], test_df["value"], label="Факт", marker="o")
    # Прогнозы всех моделей
    model_cols = [c for c in pred_df.columns if c not in ("ds", "unique_id")]
    for c in model_cols:
        plt.plot(pred_df["ds"], pred_df[c], label=c, marker="x")
    plt.title(title or "Факты vs Прогнозы")
    plt.xlabel("ds")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def rmse_series(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return sqrt(mean_squared_error(y_true, y_pred))

def calculate_rmse_by_merge(actual_df: pd.DataFrame, predict_df: pd.DataFrame) -> pd.DataFrame:
    """
    Считает RMSE по всем моделям после merge по ds.
    actual_df: ['ds','value'] (и опц. 'unique_id')
    predict_df: ['ds','unique_id', <models>...]
    """
    cols = [c for c in predict_df.columns if c not in ("ds", "unique_id")]
    merged = predict_df.merge(
        actual_df[[TIME_COL, TARGET_COL] + ([ "unique_id"] if "unique_id" in actual_df.columns else [])],
        left_on=[TIME_COL] + (["unique_id"] if "unique_id" in predict_df.columns and "unique_id" in actual_df.columns else []),
        right_on=[TIME_COL] + (["unique_id"] if "unique_id" in predict_df.columns and "unique_id" in actual_df.columns else []),
        how="inner"
    )
    if merged.empty:
        print("⚠️ Нет пересечения дат между фактами и прогнозами — RMSE не посчитан.")
        return pd.DataFrame()

    metrics = {f"{c} RMSE": rmse_series(merged[TARGET_COL].values, merged[c].values) for c in cols}
    return pd.DataFrame([metrics])

def build_models() -> list:
    """Фабрика моделей NeuralForecast."""
    nbeats = NBEATS(h=H, input_size=INPUT_SIZE, max_steps=MAX_STEPS, loss=RMSE())
    rnn = RNN(h=H, input_size=INPUT_SIZE, max_steps=MAX_STEPS, loss=RMSE())
    mlp = MLP(h=H, input_size=INPUT_SIZE, max_steps=MAX_STEPS, loss=RMSE())
    nbeatsx = NBEATSx(
        h=H, input_size=INPUT_SIZE, loss=RMSE(),
        scaler_type='robust', dropout_prob_theta=0.5,
        max_steps=MAX_STEPS, val_check_steps=10, early_stop_patience_steps=2
    )
    nhits = NHITS(
        h=H, input_size=INPUT_SIZE, loss=RMSE(),
        scaler_type='robust', dropout_prob_theta=0.5,
        max_steps=MAX_STEPS, val_check_steps=10, early_stop_patience_steps=2
    )
    timellm = TimesNet(
        h=H, input_size=INPUT_SIZE, hidden_size=16, conv_hidden_size=32,
        loss=RMSE(), scaler_type='standard', learning_rate=1e-3,
        max_steps=MAX_STEPS, val_check_steps=50, early_stop_patience_steps=2
    )
    tsmixer = TSMixer(
        h=H, input_size=INPUT_SIZE, n_series=1, n_block=3, ff_dim=4, dropout=0,
        revin=True, scaler_type='standard', loss=RMSE(), learning_rate=1e-3,
        max_steps=MAX_STEPS, val_check_steps=50, early_stop_patience_steps=2, batch_size=30
    )
    return [nbeats, nbeatsx, mlp, nhits, timellm, rnn, tsmixer]

# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================
# Чтение всех рядов, добавление unique_id
dfs = []
for uid, path in CSV_FILES.items():
    df_i = load_series(path, uid)
    dfs.append(df_i)
df = pd.concat(dfs, axis=0, ignore_index=True).sort_values(TIME_COL)

# Быстрый контроль пропусков
na_counts = df.groupby("unique_id")[TARGET_COL].apply(lambda s: s.isna().sum())
print("NaN по цехам:\n", na_counts)

# Пример: заполним пропуски для всех рядов скользящим средним (при необходимости)
df = df.groupby("unique_id", group_keys=False).apply(lambda g: fill_na_rolling_mean(g, TARGET_COL, window=2))

# Тренировочный датасет (пример: последние 344 точки по каждому id)
train_df = df.groupby("unique_id", group_keys=False).tail(344)

# ============================================================
# МОДЕЛИ И ОБУЧЕНИЕ
# ============================================================
models = build_models()
nf = NeuralForecast(models=models, freq=FREQ)

# ВНИМАНИЕ: val_size=30 — валидация на хвостовых точках тренировочного df.
# Для честной оценки на истории лучше использовать cross_validation.
nf.fit(df=train_df, val_size=VAL_SIZE, time_col=TIME_COL, target_col=TARGET_COL)

# Прогноз на горизонт H
forecasts = nf.predict().reset_index()  # ['unique_id','ds', <models>...]

# ============================================================
# ВИЗУАЛИЗАЦИЯ И МЕТРИКИ ПО КАЖДОМУ ЦЕХУ
# ============================================================
for uid in CSV_FILES.keys():
    f_uid = forecasts[forecasts["unique_id"] == uid].copy()
    y_uid = df[df["unique_id"] == uid].copy()

    # Для наглядности сравним хвост фактов и прогноз (оси времени могут не совпасть)
    plot_forecast(y_uid.tail(H), f_uid, title=f"Цех: {uid}")

    # Если у вас есть фактические данные на период прогноза — подставьте сюда df_fact_uid
    # df_fact_uid = pd.read_csv("PATH_TO_FACT.csv")  # должен иметь 'ds','value' (+ 'unique_id' опц.)
    # df_fact_uid['ds'] = pd.to_datetime(df_fact_uid['ds'])
    # df_fact_uid['unique_id'] = uid
    # metrics = calculate_rmse_by_merge(df_fact_uid, f_uid)
    # print(f"RMSE для {uid}:\n", metrics)

# ============================================================
# ПРИМЕР: ПРОГНОЗ ДЛЯ ОДНОГО РЯДА (1 цех)
# ============================================================
uid = "1 цех"
data_one = df[df["unique_id"] == uid][[TIME_COL, TARGET_COL, "unique_id"]].copy()

nf_single = NeuralForecast(models=models, freq=FREQ)
nf_single.fit(df=data_one, val_size=VAL_SIZE, time_col=TIME_COL, target_col=TARGET_COL)
forecasts_one = nf_single.predict().reset_index()

plot_forecast(data_one.tail(H), forecasts_one, title=f"Один ряд: {uid}")

# Если есть факт на период прогноза
# df_fact = pd.read_csv("PATH_TO_FACT.csv")
# df_fact['ds'] = pd.to_datetime(df_fact['ds'])
# df_fact['unique_id'] = uid
# print(calculate_rmse_by_merge(df_fact, forecasts_one))

