import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

random.seed(42)
np.random.seed(42)

# Файлы (название для таблицы, путь)
files = [
    ('Малый датасет', 'data/data_small.xml'),
    ('Средний датасет', 'data/data_medium.xml'),
    ('Большой датасет', 'data/data_large.xml'),
]
TARGET_COL = 'total_analysis_cost'
MISSING_PERCENTS = [3, 5, 10, 20, 30]

def print_stats(df, col, label='', folder='output'):
    try:
        mode_val = df[col].mode(dropna=True)
        mode_val = mode_val.iloc[0] if not mode_val.empty else np.nan
    except Exception:
        mode_val = np.nan
    stats = {
        'mean': df[col].mean(),
        'median': df[col].median(),
        'mode': mode_val,
        'count': df[col].count()
    }
    plt.figure()
    plt.hist(df[col].dropna(), bins=20)
    plt.title(f'Распределение: {label}')
    plt.xlabel(col)
    plt.ylabel('Частота')
    img_path = os.path.join(folder, f"{label.replace(' ', '_').replace(':','')}.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
    return stats

def make_missing(df, col, percent):
    df = df.copy()
    n = int(len(df) * percent / 100)
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

# Для хранения всех результатов по всем датасетам
all_results = []
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

for dataset_name, filepath in files:
    dataset_folder = os.path.join(OUTPUT_DIR, dataset_name.replace(' ', '_'))
    os.makedirs(dataset_folder, exist_ok=True)

    print(f'\n===== {dataset_name} ({filepath}) =====')
    df = pd.read_xml(filepath)
    if TARGET_COL not in df.columns:
        print(f'Столбец {TARGET_COL} не найден. Пропускаю...')
        continue

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    full_data = df[[TARGET_COL]].dropna()

    # Эталонные значения
    etalon_stats = print_stats(full_data, TARGET_COL, label=f'Эталон', folder=dataset_folder)
    etalon_mean = etalon_stats['mean']

    results = []

    for percent in MISSING_PERCENTS:
        print(f'\n--- {percent}% пропусков ---')
        # 1. Внесение пропусков
        miss_data = make_missing(full_data, TARGET_COL, percent)
        _ = print_stats(miss_data, TARGET_COL, label=f'{percent}_пропусков', folder=dataset_folder)

        # 2.1 Удаление строк
        dropna_data = miss_data.dropna()
        stats_dropna = print_stats(dropna_data, TARGET_COL, label=f'{percent}_dropna', folder=dataset_folder)

        # 2.2 Повторение предыдущего значения
        ffill_data = fill_ffill(miss_data, TARGET_COL)
        stats_ffill = print_stats(ffill_data, TARGET_COL, label=f'{percent}_ffill', folder=dataset_folder)

        # 2.3 Линейная регрессия
        linreg_data = fill_linear_regression(miss_data, TARGET_COL)
        stats_linreg = print_stats(linreg_data, TARGET_COL, label=f'{percent}_linreg', folder=dataset_folder)

        # Считаем относительную ошибку (в %)
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

    df_result = pd.DataFrame(results)
    all_results.append((dataset_name, df_result))

    # Печать красивой таблицы в консоль
    print('\n' + '='*40)
    print(f"Финальная таблица для {dataset_name}:")
    print(df_result.to_string(index=False))
    print('='*40)
    # Сохраняем таблицу в CSV именно в папку датасета
    df_result.to_csv(os.path.join(dataset_folder, "results.csv"), index=False, sep=';')

# Если хочешь итоговую таблицу по всем датасетам — объединим:
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, 'all_datasets_results.xlsx')) as writer:
    for dataset_name, df in all_results:
        df.to_excel(writer, sheet_name=dataset_name[:30], index=False)
print("\nВсе таблицы и графики разложены по папкам в 'output'.")
