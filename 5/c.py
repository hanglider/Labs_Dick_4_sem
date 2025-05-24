import sys
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QMessageBox,
    QPlainTextEdit, QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Параметры
N_SAMPLES = 100
N_FEATURES = 15

data = fetch_covtype()
X_full = pd.DataFrame(data.data[:N_SAMPLES], columns=data.feature_names)
y_true = data.target[:N_SAMPLES]

# Оставим только первые N_FEATURES признаков для анализа
X_full = X_full.iloc[:, :N_FEATURES]
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

def pearson_distance(u, v):
    if np.std(u) == 0 or np.std(v) == 0:
        return 1.0
    return 1 - np.corrcoef(u, v)[0, 1]

def single_linkage_clustering(X, n_clusters):
    dists = pdist(X, metric=pearson_distance)
    Z = linkage(dists, method='single')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels

def evaluate_rand(y_true, y_pred):
    return rand_score(y_true, y_pred)

def select_features_additive(X, y, n_features=5, n_clusters=7):
    n_total = X.shape[1]
    selected = []
    available = list(range(n_total))
    best_score = -1

    for _ in range(n_features):
        best_feat = None
        for feat in available:
            test_feats = selected + [feat]
            labels = single_linkage_clustering(X[:, test_feats], n_clusters)
            score = evaluate_rand(y, labels)
            if score > best_score:
                best_score = score
                best_feat = feat
        if best_feat is not None:
            selected.append(best_feat)
            available.remove(best_feat)
    return selected

def shuffle_features(X):
    X_shuffled = X.copy()
    np.random.shuffle(X_shuffled.T)
    return X_shuffled

def anonymize_shuffle_within_columns(X):
    X_anon = X.copy()
    for i in range(X_anon.shape[1]):
        np.random.shuffle(X_anon[:, i])
    return X_anon

class ClusterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная кластеризация")
        self.resize(1100, 800)
        self.center_window()

        font = QFont("Arial", 15)
        self.setFont(font)

        self.layout = QVBoxLayout(self)
        self.info = QLabel(f"Загружено записей: {X_full.shape[0]}\nПризнаков: {X_full.shape[1]}")
        self.info.setFont(QFont("Arial", 17, QFont.Bold))
        self.info.setStyleSheet("margin: 8px;")
        self.layout.addWidget(self.info)

        # --- Кнопки управления в две строки ---
        btn_layout1 = QHBoxLayout()
        btn_layout2 = QHBoxLayout()
        self.btn_cluster_full = QPushButton("Кластеризация (все признаки)")
        self.btn_select_features = QPushButton("Отбор признаков (Add)")
        self.btn_cluster_selected = QPushButton("Кластеризация (отобранные признаки)")
        self.btn_shuffle = QPushButton("Shuffle features (столбцы)")
        self.btn_shuffle_cols = QPushButton("Shuffle внутри столбцов")
        self.btn_cluster_anon = QPushButton("Кластеризация (аноним.)")
        self.btn_visualize = QPushButton("Визуализация")
        all_btns = [
            self.btn_cluster_full, self.btn_select_features, self.btn_cluster_selected,
            self.btn_shuffle, self.btn_shuffle_cols, self.btn_cluster_anon, self.btn_visualize
        ]
        # Две строки по 4 и 3 кнопки
        for i, btn in enumerate(all_btns):
            btn.setMinimumHeight(38)
            btn.setFont(QFont("Arial", 14))
            btn.setStyleSheet("border: 2px solid #2980b9; border-radius: 8px; margin: 5px;")
            if i < 4:
                btn_layout1.addWidget(btn)
            else:
                btn_layout2.addWidget(btn)
        self.layout.addLayout(btn_layout1)
        self.layout.addLayout(btn_layout2)

        # --- Настройка количества признаков (SpinBox) ---
        features_layout = QHBoxLayout()
        self.feature_spin = QSpinBox()
        self.feature_spin.setMinimum(2)
        self.feature_spin.setMaximum(X_full.shape[1])
        self.feature_spin.setValue(N_FEATURES)
        self.feature_spin.setFont(QFont("Arial", 13))
        features_layout.addWidget(QLabel("Количество признаков:"))
        features_layout.addWidget(self.feature_spin)
        self.layout.addLayout(features_layout)

        # --- Настройка числа отбираемых признаков (ещё один SpinBox) ---
        select_feats_layout = QHBoxLayout()
        self.selected_features_spin = QSpinBox()
        self.selected_features_spin.setMinimum(1)
        self.selected_features_spin.setMaximum(X_full.shape[1])
        self.selected_features_spin.setValue(5)
        self.selected_features_spin.setFont(QFont("Arial", 13))
        select_feats_layout.addWidget(QLabel("Число отбираемых признаков (Add):"))
        select_feats_layout.addWidget(self.selected_features_spin)
        self.layout.addLayout(select_feats_layout)

        # --- История действий (QPlainTextEdit, только для чтения) ---
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setFont(QFont("Consolas", 12))
        self.layout.addWidget(self.history)

        self.figure, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.X = X_full_scaled
        self.y = y_true
        self.selected_features = None
        self.anonymized = False
        self.labels = None
        self.n_clusters = 7
        self.n_selected_features = self.selected_features_spin.value()
        self.X_orig = X_full_scaled.copy()

        # Способ анонимизации ('shuffle_columns' или 'shuffle_within_columns')
        self.anonymization_mode = None

        # Сигналы
        self.btn_cluster_full.clicked.connect(self.cluster_full)
        self.btn_select_features.clicked.connect(self.select_features)
        self.btn_cluster_selected.clicked.connect(self.cluster_selected)
        self.btn_shuffle.clicked.connect(self.shuffle_columns)
        self.btn_shuffle_cols.clicked.connect(self.shuffle_within_columns)
        self.btn_cluster_anon.clicked.connect(self.cluster_anonymized)
        self.btn_visualize.clicked.connect(self.visualize)
        self.feature_spin.valueChanged.connect(self.change_features)
        self.selected_features_spin.valueChanged.connect(self.change_n_selected_features)

    def center_window(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().screen().rect().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def log(self, msg):
        self.history.appendPlainText(msg)

    def change_features(self):
        n = self.feature_spin.value()
        global X_full
        X_full = pd.DataFrame(data.data[:N_SAMPLES], columns=data.feature_names)
        X_full = X_full.iloc[:, :n]
        scaler = StandardScaler()
        self.X = scaler.fit_transform(X_full)
        self.X_orig = self.X.copy()
        self.selected_features = None
        self.anonymized = False
        self.labels = None
        self.anonymization_mode = None
        self.selected_features_spin.setMaximum(n)
        if self.n_selected_features > n:
            self.selected_features_spin.setValue(n)
            self.n_selected_features = n
        self.info.setText(f"Загружено записей: {X_full.shape[0]}\nПризнаков: {X_full.shape[1]}")
        self.log(f"Обновлено количество признаков: {n}. Данные пересчитаны.")

    def change_n_selected_features(self):
        self.n_selected_features = self.selected_features_spin.value()
        self.log(f"Обновлено число отбираемых признаков методом Add: {self.n_selected_features}")

    def cluster_full(self):
        self.log("Кластеризация (все признаки)...")
        QApplication.processEvents()
        self.labels = single_linkage_clustering(self.X, self.n_clusters)
        score = evaluate_rand(self.y, self.labels)
        self.log(f"Rand index (все признаки): {score:.4f}")

    def select_features(self):
        self.log(f"Отбор признаков методом Add ({self.n_selected_features})...")
        QApplication.processEvents()
        self.selected_features = select_features_additive(
            self.X, self.y, n_features=self.n_selected_features, n_clusters=self.n_clusters
        )
        feats = ", ".join([str(f) for f in self.selected_features])
        self.log(f"Отобрано признаков: {feats}")

    def cluster_selected(self):
        if self.selected_features is None:
            QMessageBox.warning(self, "Ошибка", "Сначала выполните отбор признаков.")
            return
        self.log("Кластеризация (отобранные признаки)...")
        QApplication.processEvents()
        X_sel = self.X[:, self.selected_features]
        self.labels = single_linkage_clustering(X_sel, self.n_clusters)
        score = evaluate_rand(self.y, self.labels)
        self.log(f"Rand index (отобранные признаки): {score:.4f}")

    def shuffle_columns(self):
        QApplication.processEvents()
        self.X = shuffle_features(self.X)
        self.anonymized = True
        self.anonymization_mode = 'shuffle_columns'
        self.log("Данные обезличены.")

    def shuffle_within_columns(self):
        QApplication.processEvents()
        self.X = anonymize_shuffle_within_columns(self.X)
        self.anonymized = True
        self.anonymization_mode = 'shuffle_within_columns'
        self.log("Данные обезличены.")

    def cluster_anonymized(self):
        if not self.anonymized:
            QMessageBox.warning(self, "Ошибка", "Сначала выполните анонимизацию данных.")
            return
        if self.anonymization_mode == 'shuffle_columns':
            self.log("Кластеризация (анонимизация: shuffle columns)...")
        elif self.anonymization_mode == 'shuffle_within_columns':
            self.log("Кластеризация (анонимизация: shuffle внутри столбцов)...")
        else:
            self.log("Кластеризация (анонимизированные данные)...")
        QApplication.processEvents()
        self.labels = single_linkage_clustering(self.X, self.n_clusters)
        score = evaluate_rand(self.y, self.labels)
        self.log(f"Rand index (анонимизация): {score:.4f}")

    def visualize(self):
        if self.labels is None:
            QMessageBox.warning(self, "Ошибка", "Сначала выполните кластеризацию.")
            return
        self.ax.clear()
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(self.X)
        scatter = self.ax.scatter(X_vis[:, 0], X_vis[:, 1], c=self.labels, cmap='tab10', s=17, alpha=0.7)
        self.ax.set_title("Визуализация кластеризации (PCA)", fontsize=17)
        self.ax.set_xlabel("Первая главная компонента (ось X)", fontsize=14)
        self.ax.set_ylabel("Вторая главная компонента (ось Y)", fontsize=14)
        self.figure.colorbar(scatter, ax=self.ax, label="Кластер")
        self.ax.tick_params(labelsize=12)
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ClusterApp()
    win.show()
    sys.exit(app.exec_())
