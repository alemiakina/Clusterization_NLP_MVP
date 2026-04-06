import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

import umap
from sklearn.cluster import HDBSCAN
import plotly.express as px
from scipy.spatial import ConvexHull

from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


# -----------------------
# UI
# -----------------------
st.set_page_config(layout="wide")

st.title("Кластеризация ВКР")


df = pd.read_excel("df_excel_clean.xlsx")

mode = "Только темы"

alpha = 0.8
# -----------------------
# Модели
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('ai-forever/sbert_large_nlu_ru')

model = load_model()

@st.cache_resource
def load_ruT5():
    return pipeline(
        "text2text-generation",
        model="cointegrated/rut5-base-multitask",
        tokenizer="cointegrated/rut5-base-multitask",
        max_length=32
    )

# -----------------------
# Функции
# -----------------------
RUSSIAN_STOPWORDS = [
    # --- стандартные ---
    "и","в","на","с","по","для","как","из","к","это","при","от","до","над","под",
    "без","о","об","у","же","ли","бы","то","не","да","или","а","но","за","со",

    # --- базовый научный мусор ---
    "исследование","анализ","разработка","метод","методы","система","системы",
    "подход","оценка","обоснование","изучение",

    # --- часто встречается в твоих темах ---
    "основе","примере", "использование","использования",
    "применение","применения"

    # --- мусор из формулировок ---
    "различных","различные",

    # --- мусорные сущности ---
    "ооо","г","года"
]


def truncate(text, max_words=50):
    return " ".join(str(text).split()[:max_words])

import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # схлопываем пробелы
    text = text.strip()
    return text

def get_embeddings(df, mode="Только темы", alpha=0.8):
    titles = df['thesis_topic'].fillna("").apply(clean_text).tolist()
    embeddings = model.encode(titles)
    from sklearn.preprocessing import normalize
    X = normalize(embeddings)

    return X


def reduce_dim(X):
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)

    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=8,
        min_dist=0.01,
        metric='cosine',
        random_state=42
    )

    return umap_model.fit_transform(X_pca)

def add_cluster_boundaries(fig, X, labels, color_map):
    unique_labels = set(labels)
    X_2d = X[:, :2]

    for label in unique_labels:
        if label == -1:
            continue

        label_str = str(label)

        if label_str not in color_map:
            continue

        color = color_map[label_str]
        points = X_2d[labels == label]

        if len(points) < 3:
            continue

        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.append(hull_points, [hull_points[0]], axis=0)

            # --- 1. ОБЫЧНАЯ граница ---
            fig.add_scatter(
                x=hull_points[:, 0],
                y=hull_points[:, 1],
                mode='lines',
                line=dict(width=2, color=color),
                fill='toself',
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.08)"),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
                legendgroup=f"cluster_{label}"
            )

            # --- 2. АКТИВНАЯ граница (скрытая) ---
            fig.add_scatter(
                x=hull_points[:, 0],
                y=hull_points[:, 1],
                mode='lines',
                line=dict(width=4, color=color),
                fill='toself',
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.25)"),
                opacity=0, 
                showlegend=False,
                hoverinfo="skip",
                legendgroup=f"cluster_{label}"
            )

        except:
            continue

    return fig


def get_cluster_colors(fig):
    color_map = {}

    for trace in fig.data:
        cluster_name = trace.name  # строка label
        color = trace.marker.color

        color_map[cluster_name] = color

    return color_map

# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = HDBSCAN(
        min_cluster_size=7,
        min_samples=4,
        metric='cosine'
    )
    return clusterer.fit_predict(X)

def get_top_words_per_cluster(df, labels, text_col='thesis_topic', top_n=5):
    df = df.copy()
    df['cluster'] = labels

    cluster_keywords = {}

    for cluster in sorted(df['cluster'].unique()):
        if cluster == -1:
            continue

        texts = df[df['cluster'] == cluster][text_col].dropna().tolist()

        if len(texts) < 3:
            continue

        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=RUSSIAN_STOPWORDS
        )

        X = vectorizer.fit_transform(texts)

        scores = np.asarray(X.mean(axis=0)).flatten()
        words = np.array(vectorizer.get_feature_names_out())

        top_words = words[np.argsort(scores)[::-1][:top_n]]

        cluster_keywords[cluster] = list(top_words)

    return cluster_keywords

def generate_cluster_label_ruT5(keywords, generator):
    prompt = (
        "суммаризация: "
        + ", ".join(keywords)
    )

    result = generator(
        prompt,
        max_new_tokens=20,
        do_sample=False
    )

    text = result[0]["generated_text"].strip()

    # fallback если вдруг фигня
    if len(text) < 3:
        return " ".join(keywords[:3])

    return text

#@st.cache_data
def generate_all_cluster_names(df, labels):
    generator = load_ruT5()

    cluster_keywords = get_top_words_per_cluster(df, labels)

    cluster_names = {}

    for cluster, words in cluster_keywords.items():
        cluster_names[cluster] = generate_cluster_label_ruT5(words, generator)

    return cluster_names


def fuzzy_similarity(a, b):
    return max(
        fuzz.token_sort_ratio(a, b),
        fuzz.token_set_ratio(a, b),
        fuzz.partial_ratio(a, b)
    )


def find_similar_topics_streamlit(new_text, df, embeddings, top_n=20):
    # --- 1. Эмбеддинг нового текста ---
    new_embedding = model.encode([new_text])

    # --- 2. Косинусное сходство ---
    similarities = cosine_similarity(new_embedding, embeddings).flatten()

    results = df.copy()
    results['cosine_sim'] = (similarities * 100).round(1)

    # --- 3. TOP кандидаты по cosine ---
    top_candidates = results.sort_values(
        'cosine_sim',
        ascending=False
    ).head(100).copy()

    # --- 4. Fuzzy ---
    top_candidates['fuzzy_sim'] = top_candidates['thesis_topic'].apply(
        lambda x: fuzzy_similarity(new_text, str(x))
    )

    # --- 5. Итог (как в ноутбуке)
    top_candidates['Сходство (%)'] = top_candidates[
        ['cosine_sim', 'fuzzy_sim']
    ].max(axis=1)

    # --- 6. Сортировка
    sorted_df = top_candidates.sort_values(
        'Сходство (%)',
        ascending=False
    )

    sorted_df = sorted_df[sorted_df['Сходство (%)'] >= 75]

    cols = ['thesis_topic']
    if 'year' in df.columns:
        cols.append('year')
    if 'supervisor' in df.columns:
        cols.append('supervisor')
    cols += ['Сходство (%)', 'cosine_sim', 'fuzzy_sim']
    # защита от дублей
    cols = list(dict.fromkeys(cols))
    # защита от дублей в df
    sorted_df = sorted_df.loc[:, ~sorted_df.columns.duplicated()]
    return sorted_df.head(top_n)[cols]


@st.cache_data(show_spinner=False)
def cached_embeddings(df, mode, alpha):
    return get_embeddings(df, mode, alpha)
# -----------------------
# Основной запуск
# -----------------------
if df is not None and not df.empty:

    #df = pd.read_excel(uploaded_file)

    # -----------------------
    # ЭМБЕДДИНГИ
    # -----------------------
    with st.spinner("Считаем эмбеддинги..."):
        embeddings = cached_embeddings(df, mode, alpha)

    # -----------------------
    # КЛАСТЕРИЗАЦИЯ (ПЕРВАЯ!)
    # -----------------------
    st.subheader("Кластеризация")

    if st.button("Запустить кластеризацию"):
        with st.spinner("Снижаем размерность..."):
            X_2d = reduce_dim(embeddings)

        with st.spinner("Кластеризация..."):
            labels = cluster_data(X_2d)

        with st.spinner("Генерируем названия кластеров..."):
            cluster_names = generate_all_cluster_names(df, labels)

        # СОХРАНЯЕМ
        st.session_state.X_2d = X_2d
        st.session_state.labels = labels
        st.session_state.clustered_df = df.copy()
        st.session_state.clustered_df["cluster"] = labels
        st.session_state.cluster_names = cluster_names

    # -----------------------
    # ✅ ОТОБРАЖЕНИЕ (БЕЗ ПЕРЕСЧЁТА)
    # -----------------------
    if "X_2d" in st.session_state:

        X_2d = st.session_state.X_2d
        labels = st.session_state.labels
        df_clustered = st.session_state.clustered_df

        supervisors = sorted(
            df_clustered["supervisor_code"]
            .dropna()
            .astype(str)
            .unique()
        )

        years = sorted(
            df_clustered["year"]
            .dropna()
            .astype(str)
            .unique()
        )

        # -----------------------
        # ФИЛЬТР ПО ПРЕПОДАВАТЕЛЮ
        # -----------------------

        col1, col2, col3 = st.columns([1, 2, 2])

        with col1:
            if st.button("Сброс фильтров"):
                st.session_state.selected_supervisors = []
                st.session_state.selected_years = []

        with col2:
            if "selected_supervisors" not in st.session_state:
                st.session_state.selected_supervisors = []

            selected_supervisor = st.multiselect(
                "Фильтр по преподавателю",
                options=supervisors,
                #default=st.session_state.selected_supervisors,
                key="selected_supervisors"
            )

        with col3:
            if "selected_years" not in st.session_state:
                st.session_state.selected_years = []

            selected_years = st.multiselect(
                "Фильтр по году",
                options=years,
                #default=st.session_state.selected_years,
                key="selected_years"
            )

            clusters = sorted(df_clustered["cluster"].unique())
            cluster_options = ["Все"] + [str(c) for c in clusters if c != -1]

        selected_cluster = st.selectbox("Фильтр по кластеру", cluster_options)

        # по умолчанию показываем всех
        #if "filter_mode" not in st.session_state:
        #    st.session_state.filter_mode = "all"

        # если введён преподаватель → переключаем режим
        #if selected_supervisor.strip():
        #    st.session_state.filter_mode = "supervisor"

        # -----------------------
        # ФИЛЬТРАЦИЯ ДАННЫХ
        # -----------------------

        df_display = df_clustered.copy()

        # фильтр по кластеру
        if selected_cluster != "Все":
            df_display = df_display[
                df_display["cluster"] == int(selected_cluster)
            ]

        # фильтр по преподавателю
        if selected_supervisor:
            df_display = df_display[
                df_display["supervisor_code"].isin(selected_supervisor)
            ]

        # фильтр по годам
        if selected_years:
            df_display = df_display[
                df_display["year"].astype(str).isin(selected_years)
            ]



        # --- ГРАФИК ---
        indices = df_display.index
        X_display = X_2d[indices]
        labels_display = labels[indices]

        # --- 1. Цвета кластеров (правильные!) ---
        fig_full = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            color=labels.astype(str) 
        )

        color_map = get_cluster_colors(fig_full)

        # --- 2. ФОН (все точки серые) ---
        fig = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            opacity=0.08
        )

        fig.update_traces(
            marker=dict(color="lightgray", size=5),
            showlegend=False,
            hoverinfo="skip",
            hovertemplate=None 
        )

        # --- 3. ВЫДЕЛЕННЫЕ ТОЧКИ ---
        fig = add_cluster_boundaries(fig, X_2d, labels, color_map)

        unique_clusters = sorted(set(labels_display))

        for cluster in unique_clusters:

            cluster_mask = labels_display == cluster

            X_cluster = X_display[cluster_mask]
            df_cluster_part = df_display.iloc[cluster_mask]

            color = color_map.get(str(cluster), "gray")

            cluster_label = st.session_state.get("cluster_names", {}).get(
                    cluster,
                    f"Кластер {cluster}"
                )

            fig.add_scatter(
                x=X_cluster[:, 0],
                y=X_cluster[:, 1],
                mode='markers',
                marker=dict(
                    color=color,
                    size=9,
                    line=dict(width=0.5, color='black')
                ),
                name=cluster_label,

                # добавляем group
                legendgroup=f"cluster_{cluster}",

                # ВОТ ЭТО ДАЁТ ПОДСВЕТКУ
                selected=dict(
                marker=dict(
                    size=12,
                    opacity=1
                    )
                ),
                unselected=dict(
                    marker=dict(
                        opacity=0.15
                    )
                ),

                customdata=np.stack([
                    df_cluster_part['thesis_topic'],
                    [cluster_label] * len(df_cluster_part)
                ], axis=-1),

                hovertemplate=(
                    "<b>Кластер:</b> %{customdata[1]}<br>" +
                    "<b>Тема:</b> %{customdata[0]}<extra></extra>"
                )
            )

        # --- 4. ГРАНИЦЫ (по ВСЕМ данным) ---
        #fig = add_cluster_boundaries(fig, X_2d, labels, color_map)

        # --- 5. настройки ---
        fig.update_layout(
            title="Кластеры",
            hovermode="closest"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- МЕТРИКА ---
        mask = labels != -1
        if len(set(labels[mask])) > 1:
            score = silhouette_score(X_2d[mask], labels[mask])
            st.metric("Silhouette Score", round(score, 3))
        else:
            st.warning("Недостаточно кластеров для метрики")

        # --- ТАБЛИЦА ---
        st.data_editor(
            df_display[["thesis_topic", "cluster", "supervisor_code"]],
            use_container_width=True,
            disabled=True,
            column_config={
                "thesis_topic": st.column_config.TextColumn(
                    "Тема ВКР",
                    width="large"
                ),
                "cluster": st.column_config.NumberColumn(
                    "Кластер",
                    width=50 
                ),
                "supervisor_code": st.column_config.TextColumn(
                    "Преподаватель",
                    width=80
                )
            }
        )


    # -----------------------
    #  ПРОВЕРКА ДУБЛИКАТОВ 
    # -----------------------
    st.subheader("Проверка темы на уникальность")

    new_topic = st.text_area(
        "Введите тему ВКР",
        placeholder="Например: Переработка отходов..."
    )

    top_n = st.slider("Сколько результатов показать", 5, 50, 20)

    if st.button("Проверить уникальность"):

        if new_topic.strip():

            with st.spinner("Ищем похожие темы..."):
                similar_df = find_similar_topics_streamlit(
                    new_topic,
                    df,
                    embeddings,
                    top_n
                )

            st.write("### 🔥 Похожие темы")
            st.dataframe(similar_df, use_container_width=True)

        else:
            st.warning("Введите тему")
