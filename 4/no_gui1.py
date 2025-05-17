import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

random.seed(42)
np.random.seed(42)

files = [
    ('Малый датасет', 'data/data_small.xml'),
    ('Средний датасет', 'data/data_medium.xml'),
    ('Большой датасет', 'data/data_large.xml'),
]
TARGET_COLS_NUM = ['total_analysis_cost']
TARGET_COLS_CAT = ['doctor', 'bank_card_pay_system', 'bank_card_bank', 'symptoms']
MISSING_PERCENTS = [3, 5, 10, 20, 30]

def print_stats(df, col, label='', folder='output'):
    try:
        mode_val = df[col].mode(dropna=True)
        mode_val = mode_val.iloc[0] if not mode_val.empty else np.nan
    except Exception:
        mode_val = np.nan

    stats = {
        'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
        'median': df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else None,
        'mode': mode_val,
        'count': df[col].count()
    }

    plt.figure()
    if pd.api.types.is_numeric_dtype(df[col]):
        plt.hist(df[col].dropna(), bins=20)
        plt.xlabel(col)
        plt.ylabel('Частота')
    else:
        df[col].dropna().astype(str).value_counts().plot(kind='bar')
        plt.xlabel(col)
        plt.ylabel('Частота')
    plt.title(f'Распределение: {label}')
    img_path = os.path.join(folder, f"{label.replace(' ', '_').replace(':','')}_{col}.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    return stats

def make_missing(df, col, percent):
    df = df.copy()
    n = int(len(df) * percent / 100)
    n = max(n, 1)
    missing_idx = random.sample(list(df.index), n)
    df.loc[missing_idx, col] = np.nan
    return df

def fill_ffill(df, col):
    df = df.copy()
    df[col] = df[col].ffill()
    return df

def fill_linear_regression(df, col):
    df = df.copy()
    not_null = df[col].notnull()
    X = np.array(df.index[not_null]).reshape(-1, 1)
    y = df.loc[not_null, col]
    if len(X) > 1:
        model = LinearRegression().fit(X, y)
        nulls = df.index[~not_null]
        if len(nulls) > 0:
            predicted = model.predict(np.array(nulls).reshape(-1, 1))
            df.loc[nulls, col] = predicted
    return df

def fill_mode(df, col):
    df = df.copy()
    mode_val = df[col].mode(dropna=True)
    mode_val = mode_val.iloc[0] if not mode_val.empty else np.nan
    df[col] = df[col].fillna(mode_val)
    return df

def process_column(dataset_name, df, col, col_type, dataset_folder):
    print(f"\n=== Обработка столбца: {col} ({'числовой' if col_type == 'num' else 'категориальный'}) ===")
    full_data = df[[col]].dropna()

    etalon_stats = print_stats(full_data, col, label=f'Эталон', folder=dataset_folder)
    if col_type == 'num':
        etalon_mean = etalon_stats['mean']
    else:
        etalon_mode = etalon_stats['mode']

    results = []

    for percent in MISSING_PERCENTS:
        print(f'\n--- {percent}% пропусков ---')
        miss_data = make_missing(full_data, col, percent)
        _ = print_stats(miss_data, col, label=f'{percent}_пропусков', folder=dataset_folder)

        # 1. dropna
        dropna_data = miss_data.dropna()
        stats_dropna = print_stats(dropna_data, col, label=f'{percent}_dropna', folder=dataset_folder)

        # 2. ffill
        ffill_data = fill_ffill(miss_data, col)
        stats_ffill = print_stats(ffill_data, col, label=f'{percent}_ffill', folder=dataset_folder)

        if col_type == 'num':
            # 3. linreg
            linreg_data = fill_linear_regression(miss_data, col)
            stats_linreg = print_stats(linreg_data, col, label=f'{percent}_linreg', folder=dataset_folder)
            # Относительная ошибка по среднему
            err_dropna = abs(stats_dropna['mean'] - etalon_mean) / etalon_mean * 100
            err_ffill = abs(stats_ffill['mean'] - etalon_mean) / etalon_mean * 100
            err_linreg = abs(stats_linreg['mean'] - etalon_mean) / etalon_mean * 100
            results.append({
                'Процент_пропусков': percent,
                'Среднее_эталон': round(etalon_stats['mean'], 2),
                'Dropna': round(stats_dropna['mean'], 2),
                'Dropna_ошибка%': round(err_dropna, 2),
                'Ffill': round(stats_ffill['mean'], 2),
                'Ffill_ошибка%': round(err_ffill, 2),
                'Linreg': round(stats_linreg['mean'], 2),
                'Linreg_ошибка%': round(err_linreg, 2)
            })
        else:
            # 3. mode
            mode_data = fill_mode(miss_data, col)
            stats_mode = print_stats(mode_data, col, label=f'{percent}_mode', folder=dataset_folder)
            # Ошибка по совпадению моды
            err_dropna = int(stats_dropna['mode'] != etalon_mode)
            err_ffill = int(stats_ffill['mode'] != etalon_mode)
            err_mode = int(stats_mode['mode'] != etalon_mode)
            results.append({
                'Процент_пропусков': percent,
                'Мода_эталон': etalon_stats['mode'],
                'Dropna_мода': stats_dropna['mode'],
                'Dropna_ошибка': err_dropna,
                'Ffill_мода': stats_ffill['mode'],
                'Ffill_ошибка': err_ffill,
                'Mode_мода': stats_mode['mode'],
                'Mode_ошибка': err_mode
            })

    df_result = pd.DataFrame(results)
    print('\n' + '='*40)
    print(f"Финальная таблица для {dataset_name} - {col}:")
    print(df_result.to_string(index=False))
    print('='*40)
    df_result.to_csv(os.path.join(dataset_folder, f"results_{col}.csv"), index=False, sep=';')
    return df_result

all_results = []
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

for dataset_name, filepath in files:
    dataset_folder = os.path.join(OUTPUT_DIR, dataset_name.replace(' ', '_'))
    os.makedirs(dataset_folder, exist_ok=True)

    print(f'\n===== {dataset_name} ({filepath}) =====')
    df = pd.read_xml(filepath)

    # Обрабатываем сначала числовые, потом категориальные
    for col in TARGET_COLS_NUM:
        if col in df.columns:
            res = process_column(dataset_name, df, col, 'num', dataset_folder)
            all_results.append((f"{dataset_name} — {col}", res))
    for col in TARGET_COLS_CAT:
        if col in df.columns:
            res = process_column(dataset_name, df, col, 'cat', dataset_folder)
            all_results.append((f"{dataset_name} — {col}", res))

print("\nВсе таблицы и графики разложены по папкам в 'output'.")
