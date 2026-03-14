import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import streamlit as st
from car_analysis import car_price_analysis
from diabet_analysis import diabet_analysis

st.set_page_config(page_title="ML Analyzer", layout="wide")

st.title("Анализ данных")

st.sidebar.header("Настройки")
choice = st.sidebar.selectbox("Выберите датасет:", ["Автомобили", "Диабет"])

if st.sidebar.button("Запустить полный анализ"):
    with st.spinner("Загрузка..."):
        if choice == "Автомобили":
            results = car_price_analysis(show_plots=False)
        else:
            results = diabet_analysis(show_plots=False)

    st.success("Анализ завершен успешно!")

    tab1, tab2, tab3 = st.tabs(["Статистика и Метрики", "Распределения", "Корреляция"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Описательная статистика")
            st.dataframe(results['stats_df'])
        with col2:
            st.subheader("Сравнение моделей")
            st.dataframe(results['metrics_df'])

        st.subheader("Визуальное сравнение моделей")
        st.pyplot(results['fig_comp'])

    with tab2:
        st.subheader("Гистограммы признаков")
        st.pyplot(results['fig_dist'])

    with tab3:
        st.subheader("Матрица корреляции")
        st.pyplot(results['fig_corr'])

else:
    st.info("Выберите параметры слева и нажмите кнопку запуска.")