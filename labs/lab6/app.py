import streamlit as st
import re
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords', quiet=True)
morph = MorphAnalyzer()
ru_stop = set(stopwords.words('russian'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    tokens = text.split()
    tokens = [morph.parse(w)[0].normal_form for w in tokens
              if w not in ru_stop and len(w) > 2]
    return ' '.join(tokens), tokens


def parse_textarea_input(raw_text):
    documents = []
    current_class = None
    current_text = []

    for line in raw_text.strip().split('\n'):
        line = line.strip()
        if not line:
            if current_text and current_class is not None:
                full_text = ' '.join(current_text)
                documents.append((full_text, current_class))
                current_text = []
            continue

        if line in ('0', '1'):
            if current_text and current_class is not None:
                full_text = ' '.join(current_text)
                documents.append((full_text, current_class))
                current_text = []
            current_class = int(line)
        else:
            current_text.append(line)

    if current_text and current_class is not None:
        full_text = ' '.join(current_text)
        documents.append((full_text, current_class))

    return documents


st.set_page_config(page_title="Лаб 6.1 — Песни vs Стихи", layout="wide")

st.title("Лабораторная работа 6.1 — Обработка текста и классификация")
st.markdown("**Русский язык** — песни (0) vs классические стихи (1)")

analysis_mode = st.sidebar.radio(
    "Режим анализа",
    options=[
        "Вместе (песни + стихи)",
        "Только песни (класс 0)",
        "Только стихи (класс 1)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Рекомендуется минимум 8–10 текстов на класс для осмысленных результатов Word2Vec и t-SNE")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Загрузка данных",
    "2. TF-IDF & WordCloud",
    "3. Word2Vec + t-SNE",
    "4. Классификация"
])

with tab1:
    st.subheader("Способ 1 — загрузка готового .csv")
    st.markdown("""
    Ожидаемые столбцы:  
    • `raw_text` — текст произведения  
    • `label` — 0 (песня) или 1 (стих)  
    Опционально: `source`, `class_name`
    """)

    uploaded_csv = st.file_uploader("Выберите .csv файл", type="csv")

    st.subheader("Способ 2 — ручной ввод / копи-паст")
    st.markdown("""
    Формат ввода:  
    `0` или `1` на отдельной строке — метка класса  
    следующие строки — текст (переносы сохраняются)  
    пустая строка — разделитель между произведениями
    """)

    raw_text_input = st.text_area(
        "Вставьте тексты сюда",
        height=380,
        placeholder="""1
Я помню чудное мгновенье...
...

0
Выйду ночью в поле с конём...
"""
    )

    all_data = []

    if uploaded_csv is not None:
        try:
            df_csv = pd.read_csv(uploaded_csv, encoding="utf-8-sig")
            required = {"raw_text", "label"}
            if not required.issubset(set(df_csv.columns)):
                st.error("В файле должны быть как минимум столбцы: raw_text, label")
            else:
                df_csv["label"] = df_csv["label"].astype(int)
                df_csv["class_name"] = df_csv["label"].map({0: "песня", 1: "стих"})
                df_csv["source"] = df_csv.get("source", "загруженный csv")
                records = df_csv[["raw_text", "label", "class_name", "source"]].to_records(index=False)
                all_data.extend(records)
                st.success(f"Загружено {len(df_csv)} текстов из CSV")
        except Exception as e:
            st.error(f"Ошибка при чтении CSV: {str(e)}")

    if raw_text_input.strip():
        textarea_docs = parse_textarea_input(raw_text_input)
        for text, label in textarea_docs:
            class_name = "песня" if label == 0 else "стих"
            all_data.append((text, label, class_name, "вручную"))

    if all_data:
        df = pd.DataFrame(all_data, columns=["raw_text", "label", "class_name", "source"])

        if analysis_mode == "Только песни (класс 0)":
            df = df[df["label"] == 0].copy()
        elif analysis_mode == "Только стихи (класс 1)":
            df = df[df["label"] == 1].copy()

        if len(df) == 0:
            st.warning("В выбранном режиме нет текстов")
            st.stop()

        st.success(f"После фильтрации: **{len(df)}** текстов  →  **{analysis_mode}**")

        preview_df = df.copy()
        preview_df["preview"] = preview_df["raw_text"].str.replace(r"\n", " ", regex=True).str[:100] + "..."
        st.dataframe(
            preview_df[["source", "class_name", "label", "preview"]],
            use_container_width=True,
            column_config={"preview": st.column_config.TextColumn(width="medium")}
        )

        with st.spinner("Выполняется лемматизация..."):
            df[["clean_text", "tokens"]] = df["raw_text"].apply(
                lambda x: pd.Series(preprocess_text(x))
            )

        all_tokens_flat = [t for sublist in df["tokens"] for t in sublist]

        st.session_state.df = df
        st.session_state.all_tokens = all_tokens_flat
        st.session_state.cleaned_texts = df["clean_text"].tolist()
        st.session_state.analysis_mode = analysis_mode

with tab2:
    if "df" not in st.session_state:
        st.info("Загрузите данные на первой вкладке")
    else:
        st.caption(f"Анализ: **{st.session_state.analysis_mode}**  •  {len(st.session_state.df)} текстов")

        st.subheader("TF-IDF — наиболее характерные слова")

        vec = TfidfVectorizer(max_features=400, min_df=2)
        X = vec.fit_transform(st.session_state.cleaned_texts)

        sums = X.sum(axis=0).A1
        names = vec.get_feature_names_out()
        top_words = sorted(zip(names, sums), key=lambda x: x[1], reverse=True)[:25]

        st.table(pd.DataFrame(top_words, columns=["слово", "суммарный TF-IDF вес"]))

        st.subheader("Облако слов")

        wc_text = " ".join(st.session_state.cleaned_texts)
        wc = WordCloud(
            width=1000, height=600,
            background_color="white",
            max_words=180,
            colormap="Dark2" if "стихи" in st.session_state.analysis_mode.lower() else "viridis"
        ).generate(wc_text)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

with tab3:
    if "df" not in st.session_state:
        st.info("Загрузите данные на первой вкладке")
    else:
        st.caption(f"Анализ: **{st.session_state.analysis_mode}**  •  {len(st.session_state.df)} текстов")

        st.subheader("Word2Vec — близкие слова")

        sentences = st.session_state.df["tokens"].tolist()

        if len(sentences) < 6 or len(set(w for s in sentences for w in s)) < 40:
            st.warning("Слишком мало данных для качественной модели Word2Vec")
        else:
            with st.spinner("Обучаем Word2Vec..."):
                model = Word2Vec(
                    sentences,
                    vector_size=100,
                    window=6,
                    min_count=2,
                    workers=4,
                    epochs=20
                )

            available_words = sorted(set(w for s in sentences for w in s if w in model.wv))
            if available_words:
                selected_word = st.selectbox("Выберите слово", options=available_words)
                if selected_word in model.wv:
                    similar = model.wv.most_similar(selected_word, topn=8)
                    st.write(f"**{selected_word}** наиболее близкие слова:")
                    for w, score in similar:
                        st.write(f"  • {w:18}  {score:.3f}")
                else:
                    st.info("Выбранное слово отсутствует в словаре модели")
            else:
                st.info("Нет слов, попавших в словарь модели")


        st.subheader("t-SNE — визуализация топ-слов")

        if "model" in locals() and len(available_words) >= 8:
            top_n = st.slider("Сколько самых частых слов показать", 8, 25, 15)
            word_counts = Counter(st.session_state.all_tokens)
            common_words = [w for w, _ in word_counts.most_common(60) if w in model.wv][:top_n]

            if len(common_words) >= 5:
                vectors = np.array([model.wv[w] for w in common_words])
                tsne = TSNE(n_components=2, perplexity=min(25, len(vectors)-1), random_state=42)
                emb_2d = tsne.fit_transform(vectors)

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.7)
                for i, word in enumerate(common_words):
                    ax.text(emb_2d[i, 0] + 0.08, emb_2d[i, 1], word, fontsize=10)
                ax.set_title(f"t-SNE • {st.session_state.analysis_mode}")
                st.pyplot(fig)
            else:
                st.info("Недостаточно слов для построения t-SNE")

with tab4:
    if "df" not in st.session_state:
        st.info("Загрузите данные на первой вкладке")
    elif st.session_state.analysis_mode != "Вместе (песни + стихи)":
        st.info("Классификация имеет смысл только в режиме **«Вместе (песни + стихи)»**")
    elif len(st.session_state.df) < 8:
        st.info("Слишком мало данных для обучения моделей (рекомендуется ≥8–10 текстов)")
    else:
        st.caption(f"Классификация: песни (0) vs стихи (1)  •  {len(st.session_state.df)} примеров")

        vec = TfidfVectorizer(max_features=800, min_df=2)
        X = vec.fit_transform(st.session_state.cleaned_texts)
        y = st.session_state.df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
            "Linear SVC": SVC(kernel="linear", probability=True, max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12)
        }

        results = []

        for name, clf in models.items():
            with st.spinner(f"Обучаем {name}..."):
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                acc = accuracy_score(y_test, pred)
                results.append({
                    "Модель": name,
                    "Accuracy": f"{acc:.3f}",
                    "Support (test)": len(y_test)
                })

        st.table(pd.DataFrame(results))

        best_model_name = max(results, key=lambda x: float(x["Accuracy"]))["Модель"]
        st.success(f"Лучший результат показал **{best_model_name}**")
