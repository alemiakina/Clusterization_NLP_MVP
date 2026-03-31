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

from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz


# -----------------------
# UI
# -----------------------
st.title("📊 Кластеризация ВКР")

uploaded_file = st.file_uploader("Загрузи Excel", type=["xlsx"])

mode = st.selectbox(
    "Режим эмбеддинга",
    ["Только темы", "Только описания", "Комбинированный"]
)

alpha = st.slider("Вес темы (alpha)", 0.0, 1.0, 0.7)

st.sidebar.header("Параметры модели")

pca_dim = st.sidebar.slider("PCA компоненты", 10, 200, 50)
umap_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 50, 15)
umap_min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 0.5, 0.1)
cluster_size = st.sidebar.slider("min_cluster_size", 5, 50, 10)

st.subheader("🔍 Проверка темы на дубликаты")

new_topic = st.text_area(
    "Введите тему ВКР",
    placeholder="Например: Анализ больших данных в медицине"
)

top_n = st.slider("Сколько результатов показать", 5, 50, 20)

if st.button("Проверить уникальность"):
    st.session_state.run_duplicates = True


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


def get_embeddings(df, mode, alpha):
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
    pca = PCA(n_components=pca_dim)
    X_pca = pca.fit_transform(X)

    umap_model = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric='cosine',
        random_state=42
    )

    return umap_model.fit_transform(X_pca)

# -----------------------
# Параметры HDBSCAN в sidebar
# -----------------------
cluster_min_size = st.sidebar.slider("HDBSCAN min_cluster_size", 5, 50, 10)
cluster_eps = st.sidebar.slider("HDBSCAN cluster_selection_epsilon", 0.0, 1.0, 0.0, 0.01)

# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
        metric='euclidean'
    )
    return clusterer.fit_predict(X)
    
# -----------------------
# Функция кластеризации
# -----------------------
def cluster_data(X):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cluster_min_size,
        cluster_selection_epsilon=cluster_eps,
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

    cols = ['thesis_topic']
    if 'year' in df.columns:
        cols.append('year')
    if 'supervisor' in df.columns:
        cols.append('supervisor')
    cols += ['Сходство (%)', 'cosine_sim', 'fuzzy_sim']
    return sorted_df.head(top_n)[cols]

@st.cache_data
def cached_embeddings(df, mode, alpha):
    return get_embeddings(df, mode, alpha)
# -----------------------
# Основной запуск
# -----------------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)

    # -----------------------
    # СЧИТАЕМ ЭМБЕДДИНГИ (один раз)
    # -----------------------
    with st.spinner("Считаем эмбеддинги..."):
        embeddings = cached_embeddings(df, mode, alpha)


    # -----------------------
    # 🔍 ПОИСК ДУБЛИКАТОВ (отдельно!)
    # -----------------------
    if st.session_state.get("run_duplicates") and new_topic.strip():
        with st.spinner("Ищем похожие темы..."):
            similar_df = find_similar_topics_streamlit(
                new_topic,
                df,
                embeddings,
                top_n
            )

        st.write("### 🔥 Похожие темы")
        st.dataframe(similar_df)

        st.download_button(
            "📥 Скачать результаты",
            similar_df.to_csv(index=False),
            "similar_topics.csv",
            "text/csv"
        )


    # -----------------------
    # 🚀 КЛАСТЕРИЗАЦИЯ (отдельная кнопка)
    # -----------------------
    if st.button("🚀 Запустить кластеризацию"):
        st.session_state.run_cluster = True
        if st.session_state.get("run_cluster"):

            with st.spinner("Снижаем размерность..."):
                X_2d = reduce_dim(embeddings)

            with st.spinner("Кластеризация..."):
                labels = cluster_data(X_2d)

            df['cluster'] = labels

    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    

    
    

    

        # -----------------------
        # Метрики
        # -----------------------
        mask = labels != -1

        if len(set(labels[mask])) > 1:
            score = silhouette_score(X_2d[mask], labels[mask])
            st.metric("Silhouette Score", round(score, 3))
        else:
            st.warning("Недостаточно кластеров для метрики")

        # -----------------------
        # Визуализация
        # -----------------------
        fig = px.scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            color=labels.astype(str),
            hover_data=[df['thesis_topic']],
            title="Кластеры"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------
        # Таблица
        # -----------------------
        st.dataframe(df)

        # -----------------------
        # Скачать
        # -----------------------
        st.download_button(
            "📥 Скачать результат",
            df.to_csv(index=False),
            "clusters.csv",
            "text/csv"
        )
