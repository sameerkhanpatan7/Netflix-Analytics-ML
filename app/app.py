import streamlit as st
import pandas as pd
import pickle
import os

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

div[data-testid="metric-container"] {
    background-color: #1f2937;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

div[data-testid="metric-container"] label {
    font-size: 18px;
}

div[data-testid="metric-container"] div {
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD FILES
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

@st.cache_resource
def load_data():

    df = pickle.load(
        open(
            os.path.join(MODEL_DIR, "recommender.pkl"),
            "rb"
        )
    )

    similarity = pickle.load(
        open(
            os.path.join(MODEL_DIR, "similarity.pkl"),
            "rb"
        )
    )

    return df, similarity

df, similarity = load_data()

# ==========================================
# RECOMMEND FUNCTION
# ==========================================

def recommend(movie_name):

    movie_name = movie_name.lower()

    matches = df[
        df["title"].str.lower() == movie_name
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    distances = list(
        enumerate(
            similarity[movie_index]
        )
    )

    distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in distances[1:6]:

        recommendations.append(
            (
                df.iloc[i[0]]["title"],
                round(i[1] * 100, 2)
            )
        )

    return recommendations

# ==========================================
# HERO SECTION
# ==========================================

st.title(
    "🎬 Netflix Recommendation System"
)

st.markdown("""
Discover similar Netflix movies and TV shows using Machine Learning.

This project combines:

- Machine Learning
- Recommendation Systems
- Data Analytics
- Streamlit Web Development
""")

st.markdown("---")

# ==========================================
# KPI CARDS
# ==========================================

total_titles = len(df)

movies_count = len(
    df[df["type"] == "Movie"]
)

tvshows_count = len(
    df[df["type"] == "TV Show"]
)

countries_count = (
    df["country"]
    .dropna()
    .nunique()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Titles",
        total_titles
    )

with col2:
    st.metric(
        "Movies",
        movies_count
    )

with col3:
    st.metric(
        "TV Shows",
        tvshows_count
    )

with col4:
    st.metric(
        "Countries",
        countries_count
    )

st.markdown("---")

# ==========================================
# RECOMMENDATION ENGINE
# ==========================================

st.subheader(
    "🎯 Netflix Recommendation Engine"
)

movie_name = st.selectbox(
    "🎬 Select a Netflix Title",
    sorted(df["title"].unique())
)

# ==========================================
# SELECTED CONTENT
# ==========================================

selected_movie = df[
    df["title"] == movie_name
].iloc[0]

st.markdown(
    "## 📺 Selected Content"
)

col1, col2 = st.columns(2)

with col1:

    st.write(
        f"**Type:** {selected_movie['type']}"
    )

    st.write(
        f"**Release Year:** {selected_movie['release_year']}"
    )

with col2:

    st.write(
        f"**Rating:** {selected_movie['rating']}"
    )

    st.write(
        f"**Genre:** {selected_movie['listed_in']}"
    )

st.markdown(
    "**Description:**"
)

st.write(
    selected_movie["description"]
)

st.markdown("---")

# ==========================================
# RECOMMENDATIONS
# ==========================================

if st.button(
    "🚀 Get Recommendations"
):

    results = recommend(movie_name)

    if not results:

        st.warning(
            "No recommendations found."
        )

    else:

        st.markdown(
            "## 🍿 Recommended For You"
        )

        for movie, score in results:

            movie_data = df[
                df["title"] == movie
            ].iloc[0]

            with st.container():

                st.markdown(
                    f"### 🎬 {movie}"
                )

                st.write(
                    f"🎯 Similarity Score: {score}%"
                )

                st.write(
                    f"**Type:** {movie_data['type']}"
                )

                st.write(
                    f"**Release Year:** {movie_data['release_year']}"
                )

                st.write(
                    f"**Rating:** {movie_data['rating']}"
                )

                st.write(
                    f"**Genre:** {movie_data['listed_in']}"
                )

                st.info(
                    movie_data["description"]
                )

                st.divider()

# ==========================================
# PROJECT OVERVIEW
# ==========================================

st.subheader(
    "📌 Project Overview"
)

st.write("""
### Technologies Used

- Python
- Pandas
- Scikit-Learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit

### Recommendation Factors

- Genre Similarity
- Director Similarity
- Cast Similarity
- Description Similarity

### Dataset

Netflix dataset containing over 8,800 titles.

### Features

- Content-Based Recommendation Engine
- Interactive Analytics Dashboard
- KPI Cards
- Dynamic Filters
- Dataset Explorer
""")