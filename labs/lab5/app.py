import streamlit as st
from utils.data_loader import load_and_preprocess_data
from utils.eda_component import show_eda
from utils.dim_reduction_component import run_dimensionality_reduction
from utils.clustering_component import run_clustering

st.set_page_config(page_title="Снижение размерности + Кластеризация", layout="wide")
st.title("Методы снижения размерности и кластеризация")

st.sidebar.header("Настройки")
dataset_choice = st.sidebar.selectbox(
    "Выберите датасет",
    ["Penguins (penguins.csv)", "Bike Sales (final_cleaned_data.csv)"]
)

task = st.sidebar.radio("Задача", ["EDA + Задача 1: Снижение размерности", "Задача 2: Кластеризация"])

if st.sidebar.button("Загрузить и обработать данные"):
    df, numerical_cols, target_col, dataset_name = load_and_preprocess_data(
        dataset_choice,
        session_state=st.session_state
    )
    st.session_state['df'] = df
    st.session_state['numerical_cols'] = numerical_cols
    st.session_state['target_col'] = target_col
    st.session_state['dataset_name'] = dataset_name
    st.success("Данные загружены и предобработаны!")

if 'df' in st.session_state:
    df = st.session_state['df']
    numerical_cols = st.session_state['numerical_cols']
    target_col = st.session_state['target_col']

    if task.startswith("EDA"):
        show_eda(df, target_col)
        run_dimensionality_reduction(df, numerical_cols, target_col)
    else:
        run_clustering(df, numerical_cols, target_col)
else:
    st.info("Нажмите кнопку «Загрузить и обработать данные»")