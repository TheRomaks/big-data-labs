import streamlit as st
import plotly.express as px


def show_eda(df, target_col):
    st.header("1. Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Первые 5 строк")
        st.dataframe(df.head())
    with col2:
        st.subheader("Описание данных")
        st.dataframe(df.describe())

    st.subheader("Гистограммы числовых признаков")
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if col != target_col:
            fig = px.histogram(df, x=col, color=df[target_col].astype(str), nbins=30)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Корреляционная матрица")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)