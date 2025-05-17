import sys
import os
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

# ======= Стандартный режим (как раньше) =======
def classic_mode():
    print("Режим обычного анализа без GUI")
    pass

# ======= GUI PyQt5 =======
def gui_mode():
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QFileDialog, QVBoxLayout, QPushButton, QLabel,
        QHBoxLayout, QSpinBox, QTableWidget, QTableWidgetItem, QComboBox, QMessageBox
    )
    from PyQt5.QtCore import Qt

    # Список категориальных столбцов, которые должны быть доступны для восстановления
    CATEGORY_COLS = ["doctor", "bank_card_pay_system", "bank_card_bank"]

    class DataImputerGUI(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Лабораторная: Работа с пропусками")
            self.setMinimumSize(900, 650)
            self.data = None
            self.corrupted_data = None
            self.result_data = None

            # Layouts
            main_layout = QVBoxLayout()
            file_layout = QHBoxLayout()
            controls_layout = QHBoxLayout()
            stats_layout = QVBoxLayout()

            # Файл
            self.file_label = QLabel("Файл не выбран")
            btn_load = QPushButton("Выбрать датасет")
            btn_load.clicked.connect(self.load_dataset)
            file_layout.addWidget(btn_load)
            file_layout.addWidget(self.file_label)

            # Процент пропусков
            self.percent_label = QLabel("Процент пропусков:")
            self.percent_box = QSpinBox()
            self.percent_box.setRange(1, 99)
            self.percent_box.setValue(10)

            # Колонка для анализа
            self.col_combo = QComboBox()
            self.col_combo.setMinimumWidth(180)
            self.col_combo.currentIndexChanged.connect(self.update_restore_methods)
            self.col_combo.currentIndexChanged.connect(self.refresh_stats)

            # Метод восстановления
            self.restore_method = QComboBox()
            self.restore_method.setMinimumWidth(110)

            # Кнопки
            btn_corrupt = QPushButton("Испортить (внести пропуски)")
            btn_restore = QPushButton("Восстановить")

            btn_corrupt.clicked.connect(self.corrupt_data)
            btn_restore.clicked.connect(self.restore_data)

            controls_layout.addWidget(self.percent_label)
            controls_layout.addWidget(self.percent_box)
            controls_layout.addWidget(QLabel("Столбец:"))
            controls_layout.addWidget(self.col_combo)
            controls_layout.addWidget(QLabel("Метод восстановления:"))
            controls_layout.addWidget(self.restore_method)
            controls_layout.addWidget(btn_corrupt)
            controls_layout.addWidget(btn_restore)

            # Таблица с датасетом (просмотр)
            self.table = QTableWidget()
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.table.setMinimumHeight(250)

            # Статистика
            self.stats_label = QLabel("Информация о столбце появится тут.")
            stats_layout.addWidget(self.stats_label)

            main_layout.addLayout(file_layout)
            main_layout.addLayout(controls_layout)
            main_layout.addWidget(QLabel("<b>Данные (первые строки):</b>"))
            main_layout.addWidget(self.table)
            main_layout.addLayout(stats_layout)
            self.setLayout(main_layout)

        def load_dataset(self):
            fname, _ = QFileDialog.getOpenFileName(self, "Выбрать датасет", "", "XML файлы (*.xml);;CSV (*.csv);;Все файлы (*)")
            if not fname:
                return
            try:
                if fname.endswith('.xml'):
                    self.data = pd.read_xml(fname)
                elif fname.endswith('.csv'):
                    self.data = pd.read_csv(fname)
                else:
                    raise Exception("Поддерживаются только XML и CSV")
                self.file_label.setText(f"Загружено: {os.path.basename(fname)} ({len(self.data)} строк)")
                self.corrupted_data = None
                self.result_data = None
                self.col_combo.clear()
                # Добавляем все числовые и нужные категориальные
                for col in self.data.columns:
                    if pd.api.types.is_numeric_dtype(self.data[col]) or col in CATEGORY_COLS:
                        self.col_combo.addItem(col)
                if self.col_combo.count() == 0:
                    QMessageBox.warning(self, "Ошибка", "Нет подходящих столбцов в датасете.")
                self.refresh_table(self.data)
                self.update_restore_methods()
                self.refresh_stats()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки", str(e))

        def update_restore_methods(self):
            col = self.col_combo.currentText()
            self.restore_method.clear()
            if col in CATEGORY_COLS:
                self.restore_method.addItems(["dropna", "ffill", "mode"])
            else:
                self.restore_method.addItems(["dropna", "ffill", "linreg"])

        def refresh_table(self, df):
            if df is None or len(df) == 0:
                self.table.clear()
                self.table.setRowCount(0)
                self.table.setColumnCount(0)
                return
            N = min(30, len(df))
            self.table.setColumnCount(len(df.columns))
            self.table.setRowCount(N)
            self.table.setHorizontalHeaderLabels(df.columns)
            for i in range(N):
                for j, col in enumerate(df.columns):
                    val = str(df.iloc[i, j])
                    self.table.setItem(i, j, QTableWidgetItem(val))
            self.table.resizeColumnsToContents()

        def corrupt_data(self):
            if self.data is None:
                QMessageBox.warning(self, "Внимание", "Сначала загрузите датасет.")
                return
            col = self.col_combo.currentText()
            percent = self.percent_box.value()
            self.corrupted_data = self.data.copy()
            idx = self.corrupted_data[self.corrupted_data[col].notna()].index.tolist()
            n = int(len(idx) * percent / 100)
            n = max(n, 1)
            miss_idx = random.sample(idx, n)
            self.corrupted_data.loc[miss_idx, col] = np.nan
            self.result_data = None
            self.refresh_table(self.corrupted_data)
            self.refresh_stats()
            QMessageBox.information(self, "Пропуски внесены", f"В столбце {col} теперь {percent}% пропусков.")

        def restore_data(self):
            if self.corrupted_data is None:
                QMessageBox.warning(self, "Внимание", "Внесите пропуски перед восстановлением.")
                return
            col = self.col_combo.currentText()
            method = self.restore_method.currentText()
            df = self.corrupted_data.copy()
            # Категориальные столбцы
            if col in CATEGORY_COLS:
                if method == 'dropna':
                    self.result_data = df.dropna(subset=[col])
                elif method == 'ffill':
                    self.result_data = df.copy()
                    self.result_data[col] = self.result_data[col].ffill()
                elif method == 'mode':
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                    self.result_data = df.copy()
                    self.result_data[col] = self.result_data[col].fillna(mode_val)
            # Числовые столбцы
            else:
                if method == 'dropna':
                    self.result_data = df.dropna(subset=[col])
                elif method == 'ffill':
                    self.result_data = df.copy()
                    self.result_data[col] = self.result_data[col].ffill()
                elif method == 'linreg':
                    self.result_data = self.linreg_impute(df, col)
            self.refresh_table(self.result_data)
            self.refresh_stats(result=True)

        def linreg_impute(self, df, col):
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

        def refresh_stats(self, result=False):
            col = self.col_combo.currentText()
            if result and self.result_data is not None:
                df = self.result_data
                tag = "Восстановленный"
            elif self.corrupted_data is not None:
                df = self.corrupted_data
                tag = "Испортченный"
            elif self.data is not None:
                df = self.data
                tag = "Оригинал"
            else:
                self.stats_label.setText("Нет данных для анализа.")
                return
            missing = df[col].isna().sum()
            total = len(df)
            if pd.api.types.is_numeric_dtype(df[col]):
                mean = df[col].mean()
                median = df[col].median()
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else '-'
                stats = (
                    f"<b>{tag} ({col})</b><br>"
                    f"Всего строк: <b>{total}</b><br>"
                    f"Пропусков: <b>{missing}</b> ({(missing/total*100):.1f}%)<br>"
                    f"Среднее: <b>{mean:.2f}</b><br>"
                    f"Медиана: <b>{median:.2f}</b><br>"
                    f"Мода: <b>{mode_val}</b>"
                )
            else:  # категориальный
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else '-'
                stats = (
                    f"<b>{tag} ({col})</b><br>"
                    f"Всего строк: <b>{total}</b><br>"
                    f"Пропусков: <b>{missing}</b> ({(missing/total*100):.1f}%)<br>"
                    f"Мода: <b>{mode_val}</b>"
                )
            self.stats_label.setText(stats)

    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    win = DataImputerGUI()
    win.show()
    sys.exit(app.exec_())

# ======= Выбор режима =======
def main():
    print("Выберите режим работы:")
    print("1 — Классический терминальный анализ (по методичке)")
    print("2 — Графический интерфейс (PyQt5, мышкой, красиво)")
    choice = '2'#input("Введите 1 или 2: ")
    if choice.strip() == '2':
        gui_mode()
    else:
        classic_mode()

if __name__ == "__main__":
    main()
