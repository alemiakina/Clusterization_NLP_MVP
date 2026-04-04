### ПОКА НЕ БУДЕТ ПРАВОК -- НИЧЕГО НЕ ТРОГАЮ! ВИЗУАЛ ГРАФИКА ИМХО ХОРОШ, ПРАКТИЧЕСКИ МАКСИМУМ, ЧТО Я СМОГ ВЫТЯНУТЬ ИЗ ТЕКУЩЕГО ИНСТРУМЕНТАРИЯ ###

import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

import umap
import hdbscan
import plotly.express as px
from scipy.spatial import ConvexHull

from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz


# -----------------------
# UI
# -----------------------
st.title("Кластеризация ВКР")

#uploaded_file = st.file_uploader("Загрузи Excel", type=["xlsx"])
df = pd.read_excel("df_excel_clean.xlsx")
#mode = st.selectbox(
#    "Режим эмбеддинга",
#    ["Только темы", "Только описания", "Комбинированный"]
#)
mode = "Только темы"

#alpha = st.slider("Вес темы (alpha)", 0.0, 1.0, 0.7)
alpha = 0.8

#st.sidebar.header("Параметры модели")

#pca_dim = st.sidebar.slider("PCA компоненты", 10, 200, 50)
#umap_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 50, 15)
#umap_min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 0.5, 0.1)
#cluster_size = st.sidebar.slider("min_cluster_size", 5, 50, 10)

# -----------------------
# Модель
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('ai-forever/sbert_large_nlu_ru')

model = load_model()


# -----------------------
# Функции
# -----------------------
def truncate(text, max_words=50):
    return " ".join(str(text).split()[:max_words])


def get_embeddings(df, mode="Только темы", alpha=0.8):
    titles = df['thesis_topic'].fillna("").tolist()
    descs = df['description_thesis'].fillna("").apply(truncate).tolist()

    title_emb = model.encode(titles, normalize_embeddings=True)
    desc_emb = model.encode(descs, normalize_embeddings=True)

    if mode == "Только темы":
        return title_emb

    elif mode == "Только описания":
        return desc_emb

    else:
        return alpha * title_emb + (1 - alpha) * desc_emb


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
                opacity=0,  # 🔥 СКРЫТА
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
# Параметры HDBSCAN в sidebar
# -----------------------
#cluster_min_size = st.sidebar.slider("HDBSCAN min_cluster_size", 5, 50, 10)
#cluster_eps = st.sidebar.slider("HDBSCAN cluster_selection_epsilon", 0.0, 1.0, 0.0, 0.01)

# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=7,
        min_samples=4,
        #cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)


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

    sorted_df = sorted_df[sorted_df['Сходство (%)'] >= 85]

    cols = ['thesis_topic']
    if 'year' in df.columns:
        cols.append('year')
    if 'supervisor' in df.columns:
        cols.append('supervisor')
    cols += ['Сходство (%)', 'cosine_sim', 'fuzzy_sim']
    # 🔥 защита от дублей
    cols = list(dict.fromkeys(cols))
    # 🔥 защита от дублей в df
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
    # 🚀 КЛАСТЕРИЗАЦИЯ (ПЕРВАЯ!)
    # -----------------------
    st.subheader("Кластеризация")

    if st.button("Запустить кластеризацию"):
        with st.spinner("Снижаем размерность..."):
            X_2d = reduce_dim(embeddings)

        with st.spinner("Кластеризация..."):
            labels = cluster_data(X_2d)

        # 💾 СОХРАНЯЕМ
        st.session_state.X_2d = X_2d
        st.session_state.labels = labels
        st.session_state.clustered_df = df.copy()
        st.session_state.clustered_df["cluster"] = labels

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

        # -----------------------
        # 🎛️ ФИЛЬТР ПО ПРЕПОДАВАТЕЛЮ
        # -----------------------

        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("Показать всех"):
                st.session_state.selected_supervisors = []

        with col2:
            if "selected_supervisors" not in st.session_state:
                st.session_state.selected_supervisors = []

            selected_supervisor = st.multiselect(
                "Фильтр по преподавателю",
                options=supervisors,
                default=st.session_state.selected_supervisors,
                key="selected_supervisors"
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
        # 📊 ФИЛЬТРАЦИЯ ДАННЫХ
        # -----------------------

        df_display = df_clustered.copy()

        # 🔥 фильтр по кластеру
        if selected_cluster != "Все":
            df_display = df_display[
                df_display["cluster"] == int(selected_cluster)
            ]

        # 🔥 фильтр по преподавателю
        if selected_supervisor:
            df_display = df_display[
                df_display["supervisor_code"].isin(selected_supervisor)
            ]

        # --- ГРАФИК ---
        indices = df_display.index
        X_display = X_2d[indices]
        labels_display = labels[indices]

        # --- 1. Цвета кластеров (правильные!) ---
        fig_full = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            color=labels.astype(str)  # 🔥 ВАЖНО: делаем категории
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

            fig.add_scatter(
                x=X_cluster[:, 0],
                y=X_cluster[:, 1],
                mode='markers',
                marker=dict(
                    color=color,
                    size=9,
                    line=dict(width=0.5, color='black')
                ),
                name=f"Кластер {cluster}",

                # 🔥 ВАЖНО: добавляем group
                legendgroup=f"cluster_{cluster}",

                # 🔥 ВОТ ЭТО ДАЁТ ПОДСВЕТКУ
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
                    [cluster] * len(df_cluster_part)
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
        st.dataframe(df_display[["thesis_topic", "cluster", "supervisor_code"]])

        # --- СКАЧИВАНИЕ ---
        #st.download_button(
        #    "📥 Скачать результат",
        #    df_clustered.to_csv(index=False),
        #    "clusters.csv",
        #    "text/csv"
        #)
        

    # -----------------------
    # 🔍 ПРОВЕРКА ДУБЛИКАТОВ (ПОСЛЕ ГРАФИКА!)
    # -----------------------
    st.subheader("Проверка темы на уникальность")

    new_topic = st.text_area(
        "Введите тему ВКР",
        placeholder="Например: Анализ больших данных в медицине"
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
            st.dataframe(similar_df)

            #st.download_button(
            #    "📥 Скачать результаты",
            #    similar_df.to_csv(index=False),
            #    "similar_topics.csv",
            #    "text/csv"
            #)
        else:
            st.warning("Введите тему")
