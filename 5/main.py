import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit
)
from PyQt5.QtGui import QFont
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer

def single_linkage_clustering(X, n_clusters=2):
    # Если только 1 признак - все расстояния 0, будет nan
    if X.shape[1] == 1:
        # Все объекты - расстояние 0, нельзя кластеризовать, даём фейк-кластеры
        return np.ones(X.shape[0], dtype=int)
    dist = pdist(X, lambda u, v: 1 - np.corrcoef(u, v)[0, 1])
    # Заменим nan/inf на 1 (максимальное расстояние)
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    Z = linkage(dist, method='single')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels

def add_algorithm(X, y, n_features):
    features = []
    remaining = list(range(X.shape[1]))
    best_score = -1
    for _ in range(n_features):
        best_feature = None
        for f in remaining:
            test_features = features + [f]
            labels = single_linkage_clustering(X[:, test_features], n_clusters=len(set(y)))
            score = adjusted_rand_score(y, labels)
            if score > best_score:
                best_score = score
                best_feature = f
        features.append(best_feature)
        remaining.remove(best_feature)
    return features

def anonymize_df(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

class ClusterLabApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная: Кластеризация данных (PyQt5, breast_cancer)")
        self.resize(1000, 600)
        self.df = None
        self.true_labels = None
        self.best_features = None
        self.anonymized_df = None
        self.ari_all = None
        self.ari_selected = None
        self.ari_anonym = None
        self.load_sklearn_dataset()
        self.initUI()

    def load_sklearn_dataset(self):
        data = load_breast_cancer()
        X = data['data']
        y = data['target']
        df = pd.DataFrame(X, columns=data['feature_names'])
        self.df = df
        self.true_labels = y
        print(f"Загружен датасет sklearn (breast_cancer), shape = {df.shape}")

    def initUI(self):
        layout = QVBoxLayout()

        self.load_btn = QPushButton("Датасет sklearn загружен (breast_cancer)")
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)

        self.cluster_btn = QPushButton("Кластеризация (все признаки)")
        self.cluster_btn.clicked.connect(self.run_clustering)
        layout.addWidget(self.cluster_btn)

        self.add_btn = QPushButton("Отбор информативных признаков (Add)")
        self.add_btn.clicked.connect(self.select_features)
        layout.addWidget(self.add_btn)

        self.cluster_selected_btn = QPushButton("Кластеризация по информ. признакам")
        self.cluster_selected_btn.setEnabled(False)
        self.cluster_selected_btn.clicked.connect(self.run_clustering_selected)
        layout.addWidget(self.cluster_selected_btn)

        self.anonym_btn = QPushButton("Обезличить датасет")
        self.anonym_btn.clicked.connect(self.anonymize_data)
        layout.addWidget(self.anonym_btn)

        self.cluster_anonym_btn = QPushButton("Кластеризация (обезлич.)")
        self.cluster_anonym_btn.setEnabled(False)
        self.cluster_anonym_btn.clicked.connect(self.run_clustering_anonymized)
        layout.addWidget(self.cluster_anonym_btn)

        self.textedit = QTextEdit()
        self.textedit.setReadOnly(True)
        layout.addWidget(self.textedit)

        font = QFont()
        font.setPointSize(13)
        self.textedit.setFont(font)
        for btn in [self.load_btn, self.cluster_btn, self.add_btn,
                    self.cluster_selected_btn, self.anonym_btn, self.cluster_anonym_btn]:
            btn.setFont(font)

        self.setLayout(layout)
        self.textedit.append(f"Загружен встроенный датасет sklearn: breast_cancer\nРазмер: {self.df.shape}\n")

    def run_clustering(self):
        X = self.df.values
        n_clusters = len(set(self.true_labels))
        labels = single_linkage_clustering(X, n_clusters=n_clusters)
        score = adjusted_rand_score(self.true_labels, labels)
        self.ari_all = score
        self.labels_all = labels
        self.textedit.append(f"\nКластеризация по всем признакам:\nRand Index: {score:.3f}")
        self.cluster_selected_btn.setEnabled(True)

    def select_features(self):
        X = self.df.values
        n_feat = min(5, X.shape[1])
        features = add_algorithm(X, self.true_labels, n_feat)
        self.best_features = features
        feature_names = self.df.columns[features]
        self.textedit.append(f"\nНаиболее информативные признаки (Add):\n{list(feature_names)}")
        self.cluster_selected_btn.setEnabled(True)

    def run_clustering_selected(self):
        if self.best_features is None:
            self.textedit.append("\nСначала выберите признаки (Add)!")
            return
        X = self.df.values[:, self.best_features]
        n_clusters = len(set(self.true_labels))
        labels = single_linkage_clustering(X, n_clusters=n_clusters)
        score = adjusted_rand_score(self.true_labels, labels)
        self.ari_selected = score
        self.labels_selected = labels
        self.textedit.append(f"\nКластеризация по выбранным признакам:\nRand Index: {score:.3f}")
        self.cluster_anonym_btn.setEnabled(True)

    def anonymize_data(self):
        self.anonymized_df = anonymize_df(self.df)
        self.textedit.append("\nДатасет обезличен.")

    def run_clustering_anonymized(self):
        if self.anonymized_df is None:
            self.textedit.append("\nСначала обезличьте датасет.")
            return
        X = self.anonymized_df.values
        n_clusters = len(set(self.true_labels))
        labels = single_linkage_clustering(X, n_clusters=n_clusters)
        score = adjusted_rand_score(self.true_labels, labels)
        self.ari_anonym = score
        self.labels_anonym = labels
        self.textedit.append(f"\nКластеризация по обезличенному датасету:\nRand Index: {score:.3f}")
        self.show_summary()

    def show_summary(self):
        self.textedit.append("\n=== Сравнение результатов ===")
        self.textedit.append(f"Rand Index (все признаки): {self.ari_all if self.ari_all is not None else '---'}")
        self.textedit.append(f"Rand Index (выбранные признаки): {self.ari_selected if self.ari_selected is not None else '---'}")
        self.textedit.append(f"Rand Index (обезлич.): {self.ari_anonym if self.ari_anonym is not None else '---'}")
        self.textedit.append("Сделайте выводы: какие признаки/методы лучше для кластеризации на данном наборе данных?\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClusterLabApp()
    win.show()
    sys.exit(app.exec_())
