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
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

morph = MorphAnalyzer()

extra_stop_ru = {
    "я", "ты", "вы", "мы", "он", "она", "оно", "они", "его", "её", "их",
    "мне", "тебе", "вам", "нам", "мой", "твоя", "твой", "ваш", "наш",
    "это", "тот", "та", "то", "те", "эти", "который", "которая", "которое", "которые"
}
ru_stop = set(stopwords.words('russian')).union(extra_stop_ru)


def preprocess_text(text, lang="ru"):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    if lang == "ru":
        text = re.sub(r'[^а-яё\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [morph.parse(w)[0].normal_form
                  for w in tokens if w not in ru_stop and len(w) > 2]
    else:
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w)
                  for w in text.split() if w not in stopwords.words('english') and len(w) > 2]

    return ' '.join(tokens), tokens


def parse_textarea_input(raw_text):
    documents = []
    current_class = None
    current_text = []

    for line in raw_text.splitlines():
        stripped = line.strip()

        # Если встречаем метку класса, сохраняем предыдущий текст и начинаем новый
        if stripped in ('0', '1'):
            if current_text and current_class is not None:
                full_text = ' '.join(current_text).strip()
                if full_text:
                    documents.append((full_text, current_class))
            current_class = int(stripped)
            current_text = []
        else:
            # Накапливаем строки текста.
            # Игнорируем пустые строки (строфы), они больше не разрывают документ.
            if stripped:
                current_text.append(stripped)

    # Не забываем сохранить последний накопленный документ после завершения цикла
    if current_text and current_class is not None:
        full_text = ' '.join(current_text).strip()
        if full_text:
            documents.append((full_text, current_class))

    return documents

st.set_page_config(page_title="Лаб 6 — Песни vs Стихи", layout="wide")

st.title("Лабораторная работа 6 — Обработка текста и классификация")
st.markdown("**Песни (0) vs Стихи (1)** — Русский + Английский")

language = st.sidebar.selectbox("Язык данных", ["Русский", "Английский"], index=0)
analysis_mode = st.sidebar.radio(
    "Режим анализа",
    ["Вместе (песни + стихи)", "Только песни (класс 0)", "Только стихи (класс 1)"],
    index=0
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Загрузка данных", "2. TF-IDF & WordCloud",
    "3. Word2Vec + t-SNE", "4. Классификация"
])

with tab1:
    st.subheader("Способ 1 — загрузка .txt файлов")
    uploaded_files = st.file_uploader("Загрузите .txt файлы", type="txt", accept_multiple_files=True)

    st.subheader("Способ 2 — вставка текстов")
    raw_text_input = st.text_area("Вставьте тексты", height=380,
        placeholder="""1\nЯ помню чудное мгновенье...\n\n0\nВыйду ночью в поле с конём...""")

    all_data = []

    if uploaded_files:
        for file in sorted(uploaded_files, key=lambda x: x.name):
            try:
                text = file.read().decode('utf-8-sig')
                lower_name = file.name.lower()
                label = 0 if any(k in lower_name for k in ['песн', 'song', 'track', '0']) else 1
                class_name = "песня" if label == 0 else "стих"
                all_data.append((text, label, class_name, file.name))
            except Exception as e:
                st.warning(f"Ошибка чтения {file.name}")

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

        df = df.sort_values(by="raw_text").reset_index(drop=True)

        if len(df) == 0:
            st.warning("Нет текстов в выбранном режиме")
            st.stop()

        st.success(f"Загружено **{len(df)}** текстов → **{analysis_mode}** ({language})")

        lang_code = "ru" if language == "Русский" else "en"
        with st.spinner("Лемматизация..."):
            df[["clean_text", "tokens"]] = df["raw_text"].apply(
                lambda x: pd.Series(preprocess_text(x, lang_code))
            )

        if st.checkbox("Показать очищенные тексты (debug)", value=False):
            st.dataframe(df[["class_name", "clean_text"]].style.set_properties(**{'white-space': 'pre-wrap'}))

        st.session_state.df = df
        st.session_state.cleaned_texts = df["clean_text"].tolist()
        st.session_state.tokens_list = df["tokens"].tolist()
        st.session_state.all_tokens = [t for sublist in df["tokens"] for t in sublist]
        st.session_state.analysis_mode = analysis_mode
        st.session_state.language = language

with tab2:
    if "df" not in st.session_state:
        st.info("Загрузите данные на первой вкладке")
    else:
        st.caption(f"Анализ: **{st.session_state.analysis_mode}** • {len(st.session_state.df)} текстов")

        n_docs = len(st.session_state.cleaned_texts)
        min_df_val = max(1, min(2, n_docs // 3))

        vec = TfidfVectorizer(max_features=500, min_df=min_df_val, max_df=0.95)
        X = vec.fit_transform(st.session_state.cleaned_texts)

        sums = X.sum(axis=0).A1
        names = vec.get_feature_names_out()
        top_words = sorted(zip(names, sums), key=lambda x: x[1], reverse=True)[:25]

        st.subheader("Топ-25 слов по TF-IDF")
        st.table(pd.DataFrame(top_words, columns=["Слово", "Вес TF-IDF"]))

        st.subheader("Облако слов")
        wc_text = " ".join(st.session_state.cleaned_texts)
        wc = WordCloud(width=1000, height=600, background_color="white", max_words=180).generate(wc_text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

with tab3:
    if "df" not in st.session_state:
        st.info("Загрузите данные")
    else:
        sentences = st.session_state.tokens_list

        if len(sentences) < 6:
            st.warning("Недостаточно данных для Word2Vec")
        else:
            with st.spinner("Обучаем Word2Vec..."):
                model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=20)

            available_words = sorted(set(w for s in sentences for w in s if w in model.wv))
            if available_words:
                selected_word = st.selectbox("Выберите слово", options=available_words)
                if selected_word in model.wv:
                    similar = model.wv.most_similar(selected_word, topn=8)
                    st.write(f"**{selected_word}** — похожие слова:")
                    for w, score in similar:
                        st.write(f"  • {w:15} {score:.3f}")

            st.subheader("t-SNE визуализация")
            word_counts = Counter(st.session_state.all_tokens)
            common_words = [w for w, _ in word_counts.most_common(30) if w in model.wv][:15]

            if len(common_words) >= 5:
                vectors = np.array([model.wv[w] for w in common_words])
                tsne = TSNE(n_components=2, perplexity=min(15, len(vectors)-1), random_state=42)
                emb_2d = tsne.fit_transform(vectors)

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.7)
                for i, word in enumerate(common_words):
                    ax.text(emb_2d[i, 0] + 0.1, emb_2d[i, 1], word, fontsize=10)
                st.pyplot(fig)
            else:
                st.info("Недостаточно уникальных слов для t-SNE")


with tab4:
    if "df" not in st.session_state or st.session_state.analysis_mode != "Вместе (песни + стихи)":
        st.info("Классификация доступна только в режиме «Вместе»")
    elif len(st.session_state.df) < 6:
        st.info("Недостаточно данных")
    else:
        st.caption(f"Классификация • {len(st.session_state.df)} текстов")

        vec = TfidfVectorizer(max_features=700, min_df=1, max_df=0.95)
        X = vec.fit_transform(st.session_state.cleaned_texts)
        y = st.session_state.df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=4000, solver='liblinear', random_state=42, C=1.0),
            "Linear SVC": SVC(kernel="linear", probability=True, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
        }

        results = []
        for name, clf in models.items():
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test, pred)
            results.append({"Модель": name, "Accuracy": f"{acc:.3f}"})

        st.table(pd.DataFrame(results))
        best = max(results, key=lambda x: float(x["Accuracy"]))["Модель"]
        st.success(f"**Лучшая модель:** {best}")