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
N_FEATURES = 54

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

def select_features_additive_sample(X, y, n_features=5, n_clusters=7, sample_size=25):
    # Быстрое добавление: на случайной подвыборке
    idx = np.random.choice(X.shape[0], min(sample_size, X.shape[0]), replace=False)
    X_small = X[idx]
    y_small = y[idx]
    n_total = X.shape[1]
    selected = []
    available = list(range(n_total))
    best_score = -1

    for _ in range(n_features):
        best_feat = None
        for feat in available:
            test_feats = selected + [feat]
            labels = single_linkage_clustering(X_small[:, test_feats], n_clusters)
            score = evaluate_rand(y_small, labels)
            if score > best_score:
                best_score = score
                best_feat = feat
        if best_feat is not None:
            selected.append(best_feat)
            available.remove(best_feat)
    return selected

# def shuffle_features(X):
#     X_shuffled = X.copy()
#     np.random.shuffle(X_shuffled.T)
#     return X_shuffled
def shuffle_features(X):
    X_anon = X.copy()
    for i in range(X_anon.shape[1]):
        np.random.shuffle(X_anon[:, i])
    return X_anon

class ClusterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная кластеризация")
        self.resize(1200, 1050)
        self.center_window()

        font = QFont("Arial", 15)
        self.setFont(font)

        self.layout = QVBoxLayout(self)
        self.info = QLabel(f"Загружено записей: {X_full.shape[0]}\nПризнаков: {X_full.shape[1]}")
        self.info.setFont(QFont("Arial", 17, QFont.Bold))
        self.info.setStyleSheet("margin: 8px;")
        self.layout.addWidget(self.info)

        # История действий (QPlainTextEdit)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setFont(QFont("Consolas", 12))
        self.layout.addWidget(self.history)

        # 3 графика — холст matplotlib (делаем Figure с 3 subplot)
        self.figure, self.axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.X = X_full_scaled
        self.y = y_true
        self.selected_features = None
        self.labels_all = None
        self.labels_selected = None
        self.labels_anon = None
        self.n_clusters = 7
        self.n_selected_features = 5
        self.X_orig = X_full_scaled.copy()

        # Автоматический тест при запуске
        self.run_full_test()

    def center_window(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().screen().rect().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def log(self, msg):
        self.history.appendPlainText(msg)

    def run_full_test(self):
        self.log("Тест 1: Кластеризация по всем признакам...")
        QApplication.processEvents()
        self.labels_all = single_linkage_clustering(self.X, self.n_clusters)
        score_all = evaluate_rand(self.y, self.labels_all)
        self.log(f"Rand index (все признаки): {score_all:.4f}\n")

        self.log("Тест 2: Отбор признаков (Add, быстрая версия на подвыборке)...")
        QApplication.processEvents()
        self.selected_features = select_features_additive_sample(
            self.X, self.y, n_features=self.n_selected_features, n_clusters=self.n_clusters, sample_size=25
        )
        feats = ", ".join([str(f) for f in self.selected_features])
        self.log(f"Отобрано признаков: {feats}")
        X_sel = self.X[:, self.selected_features]
        self.labels_selected = single_linkage_clustering(X_sel, self.n_clusters)
        score_sel = evaluate_rand(self.y, self.labels_selected)
        self.log(f"Rand index (отобранные признаки): {score_sel:.4f}\n")

        self.log("Тест 3: Кластеризация по анонимизированным данным (shuffle по столбцам)...")
        QApplication.processEvents()
        X_anon = shuffle_features(self.X)
        self.labels_anon = single_linkage_clustering(X_anon, self.n_clusters)
        score_anon = evaluate_rand(self.y, self.labels_anon)
        self.log(f"Rand index (анонимизация): {score_anon:.4f}\n")

        self.visualize_all()

    def visualize_all(self):
        # 3 PCA-графика в одной фигуре
        self.figure.clf()
        axes = self.figure.subplots(1, 3)

        # 1 — все признаки
        pca1 = PCA(n_components=2)
        X_vis1 = pca1.fit_transform(self.X)
        scatter1 = axes[0].scatter(X_vis1[:, 0], X_vis1[:, 1], c=self.labels_all, cmap='tab10', s=17, alpha=0.7)
        axes[0].set_title("Кластеризация (все признаки)", fontsize=15)
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        self.figure.colorbar(scatter1, ax=axes[0], label="Кластер")

        # 2 — только отобранные признаки
        X_sel = self.X[:, self.selected_features]
        pca2 = PCA(n_components=2)
        X_vis2 = pca2.fit_transform(X_sel)
        scatter2 = axes[1].scatter(X_vis2[:, 0], X_vis2[:, 1], c=self.labels_selected, cmap='tab10', s=17, alpha=0.7)
        axes[1].set_title("Кластеризация (Add признаки)", fontsize=15)
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        self.figure.colorbar(scatter2, ax=axes[1], label="Кластер")

        # 3 — анонимизированные признаки
        pca3 = PCA(n_components=2)
        X_anon = shuffle_features(self.X)
        X_vis3 = pca3.fit_transform(X_anon)
        scatter3 = axes[2].scatter(X_vis3[:, 0], X_vis3[:, 1], c=self.labels_anon, cmap='tab10', s=17, alpha=0.7)
        axes[2].set_title("Кластеризация (shuffle столбцов)", fontsize=15)
        axes[2].set_xlabel("PC1")
        axes[2].set_ylabel("PC2")
        self.figure.colorbar(scatter3, ax=axes[2], label="Кластер")

        for ax in axes:
            ax.tick_params(labelsize=12)
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ClusterApp()
    win.show()
    sys.exit(app.exec_())
