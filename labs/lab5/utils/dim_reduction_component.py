import streamlit as st
import plotly.express as px
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap
import joblib
import numpy as np


def run_dimensionality_reduction(df, numerical_cols, target_col):
    st.header("Задача 1: Снижение размерности")

    X = df[numerical_cols].values
    y = df[target_col].values

    st.subheader("KernelPCA (все ядра)")
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    for kernel in kernels:
        kpca = KernelPCA(n_components=2, kernel=kernel, random_state=42)
        X_kpca = kpca.fit_transform(X)

        fig = px.scatter(
            x=X_kpca[:, 0], y=X_kpca[:, 1],
            color=y.astype(str),
            title=f"KernelPCA — ядро: {kernel}",
            labels={'color': target_col}
        )
        st.plotly_chart(fig, use_container_width=True)

        if kernel == 'linear':
            eigenvalues = kpca.eigenvalues_
            if eigenvalues is not None and len(eigenvalues) > 0:
                eigenvalues = eigenvalues[eigenvalues > 0]
                if len(eigenvalues) > 0:
                    explained = eigenvalues / np.sum(eigenvalues)
                    lost_variance = 1 - explained.sum()
                    st.write(f"**Дисперсия (linear):** {explained.round(4)}")
                    st.write(f"**Потерянная дисперсия:** {lost_variance:.4f}")

    n_classes = len(np.unique(y))
    if n_classes > 1:
        st.subheader("Linear Discriminant Analysis (LDA)")
        max_lda_comp = min(n_classes - 1, X.shape[1])
        n_lda_comp = st.slider("LDA n_components", 1, max_lda_comp, min(2, max_lda_comp))

        lda = LinearDiscriminantAnalysis(n_components=n_lda_comp)
        X_lda = lda.fit_transform(X, y)

        if n_lda_comp == 1:
            fig = px.scatter(x=X_lda[:, 0], y=[0] * len(X_lda), color=y.astype(str),
                             title=f"LDA ({n_lda_comp} компонент)", labels={'x': 'LD1'})
        else:
            fig = px.scatter(x=X_lda[:, 0], y=X_lda[:, 1], color=y.astype(str),
                             title=f"LDA ({n_lda_comp} компонентов)",
                             labels={'x': 'LD1', 'y': 'LD2'})
        st.plotly_chart(fig, use_container_width=True)

        if hasattr(lda, 'explained_variance_ratio_'):
            explained = lda.explained_variance_ratio_
            st.write(f"Объяснённая дисперсия: {explained.round(3)}")

    st.subheader("Сравнение t-SNE и UMAP")
    col1, col2 = st.columns(2)
    with col1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) // 3))
        X_tsne = tsne.fit_transform(X)
        fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y.astype(str), title="t-SNE")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X) - 1))
        X_umap = reducer.fit_transform(X)
        fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], color=y.astype(str), title="UMAP")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Сохранить лучшую модель (KernelPCA rbf)"):
        best_kpca = KernelPCA(n_components=2, kernel='rbf', random_state=42)
        best_kpca.fit(X)
        joblib.dump(best_kpca, "best_kernelpca_rbf.pkl")
        st.success("Модель сохранена: best_kernelpca_rbf.pkl")

    if st.button("Загрузить модель"):
        try:
            model = joblib.load("best_kernelpca_rbf.pkl")
            st.success("Модель загружена!")
        except FileNotFoundError:
            st.error("Файл модели не найден. Сначала сохраните модель.")