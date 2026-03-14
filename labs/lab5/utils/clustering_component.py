import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kmeans_manual(X, n_clusters, max_iters=300, tol=1e-4, random_state=42):
    np.random.seed(random_state)
    indices = np.random.choice(len(X), n_clusters, replace=False)
    centroids = X[indices].copy()

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return labels, centroids


def run_clustering(df, numerical_cols, target_col):
    st.header("Задача 2: Кластеризация")
    X = df[numerical_cols].values
    y_true = df[target_col].values

    st.subheader("Определение оптимального числа кластеров")

    col1, col2 = st.columns(2)
    inertias, silhouettes = [], []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    with col1:
        fig_elbow = px.line(x=list(range(2, 11)), y=inertias, markers=True,
                            title="Метод локтя (Inertia)", labels={'x': 'k', 'y': 'Inertia'})
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        fig_silhouette = px.line(x=list(range(2, 11)), y=silhouettes, markers=True,
                                 title="Метод силуэта", labels={'x': 'k', 'y': 'Silhouette Score'},
                                 color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig_silhouette, use_container_width=True)

    best_k_silhouette = list(range(2, 11))[np.argmax(silhouettes)]
    st.info(f"По методу силуэта оптимальное k = {best_k_silhouette} (score = {max(silhouettes):.3f})")

    n_clusters = st.slider("Выберите число кластеров", 2, 10, best_k_silhouette)

    #K-means (sklearn)
    st.subheader("K-means (sklearn)")
    kmeans_sklearn = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters_sklearn = kmeans_sklearn.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    target_labels = y_true.astype(str)
    if 'label_encoder' in st.session_state:
        try:
            known_labels = set(st.session_state['label_encoder'].classes_)
            if set(y_true).issubset(known_labels):
                y_labels = st.session_state['label_encoder'].inverse_transform(y_true)
                target_labels = y_labels.astype(str)
        except (ValueError, AttributeError, KeyError):
            pass

    df_plot = pd.DataFrame(X_2d, columns=['PC1', 'PC2'])
    df_plot['cluster'] = clusters_sklearn.astype(str)
    df_plot[target_col] = target_labels

    fig = px.scatter(df_plot, x='PC1', y='PC2', color='cluster', symbol=target_col,
                     title=f"K-means кластеры (k={n_clusters})", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.write("Метрики качества кластеризации:")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        sil = silhouette_score(X, clusters_sklearn)
        st.metric("Silhouette Score", f"{sil:.3f}")
    with col_m2:
        dbi = davies_bouldin_score(X, clusters_sklearn)
        st.metric("Davies-Bouldin Index", f"{dbi:.3f}")
    with col_m3:
        if len(np.unique(y_true)) > 1:
            ari = adjusted_rand_score(y_true, clusters_sklearn)
            st.metric("Adjusted Rand Index (vs target)", f"{ari:.3f}")

    if st.button("Сохранить модель K-means"):
        model_bundle = {
            'model': kmeans_sklearn,
            'numerical_cols': numerical_cols,
            'n_clusters': n_clusters,
            'pca': pca
        }
        joblib.dump(model_bundle, "kmeans_model.pkl")
        st.success("K-means модель сохранена!")

    st.subheader("Сравнение: собственная реализация vs sklearn")

    if st.checkbox("Запустить сравнение с ручной реализацией K-means"):
        clusters_manual, centroids_manual = kmeans_manual(X, n_clusters, random_state=42)

        col_cmp1, col_cmp2 = st.columns(2)
        with col_cmp1:
            st.write("**sklearn K-means**")
            st.write(f"- Silhouette: {silhouette_score(X, clusters_sklearn):.3f}")
            st.write(f"- Davies-Bouldin: {davies_bouldin_score(X, clusters_sklearn):.3f}")
        with col_cmp2:
            st.write("**Manual K-means**")
            st.write(f"- Silhouette: {silhouette_score(X, clusters_manual):.3f}")
            st.write(f"- Davies-Bouldin: {davies_bouldin_score(X, clusters_manual):.3f}")

        df_manual = pd.DataFrame(X_2d, columns=['PC1', 'PC2'])
        df_manual['cluster'] = clusters_manual.astype(str)
        df_manual[target_col] = target_labels

        fig_manual = px.scatter(df_manual, x='PC1', y='PC2', color='cluster', symbol=target_col,
                                title=f"Manual K-means (k={n_clusters})", height=400)
        st.plotly_chart(fig_manual, use_container_width=True)

        ari_cmp = adjusted_rand_score(clusters_sklearn, clusters_manual)
        st.info(f"Согласованность кластеризаций (ARI): {ari_cmp:.3f}")
        if ari_cmp > 0.9:
            st.success("Реализации дают практически идентичные результаты!")
        elif ari_cmp > 0.7:
            st.warning("Есть небольшие различия в назначении кластеров")
        else:
            st.error("Результаты существенно различаются")

    st.subheader("Иерархическая кластеризация")

    show_dendro = st.checkbox("Показать дендрограмму", value=False)
    if show_dendro:
        sample_size = min(len(X), 150)
        X_sample = X[:sample_size]

        try:
            Z = linkage(X_sample, method='ward')
            plt.figure(figsize=(10, 5))
            dendrogram(Z, truncate_mode='lastp', p=30)
            plt.title("Дендрограмма (выборка)")
            plt.xlabel("Индекс образца")
            plt.ylabel("Расстояние")
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.warning(f"Не удалось построить дендрограмму: {e}")

    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_clusters = agg.fit_predict(X)

    df_agg = pd.DataFrame(X_2d, columns=['PC1', 'PC2'])
    df_agg['cluster'] = agg_clusters.astype(str)
    df_agg[target_col] = target_labels

    fig_agg = px.scatter(df_agg, x='PC1', y='PC2', color='cluster', symbol=target_col,
                         title=f"Иерархическая кластеризация (k={n_clusters})", height=500)
    st.plotly_chart(fig_agg, use_container_width=True)

    st.write("Метрики для иерархической кластеризации:")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.metric("Silhouette Score", f"{silhouette_score(X, agg_clusters):.3f}")
    with col_h2:
        st.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X, agg_clusters):.3f}")

    if st.button("Сохранить модель иерархической кластеризации"):
        model_bundle = {'model': agg, 'numerical_cols': numerical_cols, 'n_clusters': n_clusters}
        joblib.dump(model_bundle, "hierarchical_model.pkl")
        st.success("Иерархическая модель сохранена!")

    st.subheader("Сравнение результатов")
    comparison_data = {
        'Метод': ['K-means (sklearn)', 'Иерархическая'],
        'Silhouette ↑': [
            f"{silhouette_score(X, clusters_sklearn):.3f}",
            f"{silhouette_score(X, agg_clusters):.3f}"
        ],
        'Davies-Bouldin ↓': [
            f"{davies_bouldin_score(X, clusters_sklearn):.3f}",
            f"{davies_bouldin_score(X, agg_clusters):.3f}"
        ]
    }
    if 'clusters_manual' in locals():
        comparison_data['Метод'].insert(1, 'K-means (manual)')
        comparison_data['Silhouette ↑'].insert(1, f"{silhouette_score(X, clusters_manual):.3f}")
        comparison_data['Davies-Bouldin ↓'].insert(1, f"{davies_bouldin_score(X, clusters_manual):.3f}")

    if len(np.unique(y_true)) > 1:
        comparison_data['ARI vs target ↑'] = [
            f"{adjusted_rand_score(y_true, clusters_sklearn):.3f}",
            f"{adjusted_rand_score(y_true, agg_clusters):.3f}"
        ]
        if 'clusters_manual' in locals():
            comparison_data['ARI vs target ↑'].insert(1, f"{adjusted_rand_score(y_true, clusters_manual):.3f}")

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)