import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from pathlib import Path

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

div[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="metric-container"] label {
    font-size: 15px;
}

div[data-testid="metric-container"] div {
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "model"

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_resource
def load_data():
    return pickle.load(
        open(MODEL_DIR / "recommender.pkl", "rb")
    )

df = load_data()

# ==========================================
# DATA PREPROCESSING
# ==========================================

df = df.copy()

df["main_genre"] = df["listed_in"].apply(
    lambda x: x.split(",")[0].strip()
    if pd.notna(x) else "Unknown"
)

# ==========================================
# TITLE
# ==========================================

st.title("📊 Netflix Analytics Dashboard")
st.caption(
    "Interactive dashboard analyzing 8,800+ Netflix titles"
)

st.markdown("---")

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("🎯 Filters")

# Type

selected_type = st.sidebar.multiselect(
    "Content Type",
    options=sorted(df["type"].dropna().unique()),
    default=sorted(df["type"].dropna().unique())
)

# Country

countries = sorted(
    df["country"]
    .dropna()
    .unique()
)

selected_country = st.sidebar.selectbox(
    "Country",
    ["All"] + countries
)

# Genre

genres = sorted(
    df["main_genre"]
    .dropna()
    .unique()
)

selected_genre = st.sidebar.selectbox(
    "Genre",
    ["All"] + genres
)

# Year

year_range = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (
        int(df["release_year"].min()),
        int(df["release_year"].max())
    )
)

filtered_df = df[
    (df["type"].isin(selected_type))
    &
    (df["release_year"] >= year_range[0])
    &
    (df["release_year"] <= year_range[1])
]

# Country Filter

if selected_country != "All":

    filtered_df = filtered_df[
        filtered_df["country"] == selected_country
    ]

# Genre Filter

if selected_genre != "All":

    filtered_df = filtered_df[
        filtered_df["main_genre"] == selected_genre
    ]
# ==========================================
# KPI CARDS
# ==========================================

total_titles = len(filtered_df)

movies_count = len(
    filtered_df[filtered_df["type"] == "Movie"]
)

tvshows_count = len(
    filtered_df[filtered_df["type"] == "TV Show"]
)

countries_count = (
    filtered_df["country"]
    .dropna()
    .nunique()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Titles", total_titles)
c2.metric("Movies", movies_count)
c3.metric("TV Shows", tvshows_count)
c4.metric("Countries", countries_count)

st.markdown("---")

# ==========================================
# KPI CARDS
# ==========================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Titles", total_titles)
c2.metric("Movies", movies_count)
c3.metric("TV Shows", tvshows_count)
c4.metric("Countries", countries_count)

# ==========================================
# KEY INSIGHTS
# ==========================================

st.markdown("### 📌 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:

    top_country = (
        filtered_df["country"]
        .mode()[0]
        if not filtered_df.empty
        else "N/A"
    )

    st.info(
        f"🌍 Top Country: {top_country}"
    )

with col2:

    top_genre = (
        filtered_df["main_genre"]
        .mode()[0]
        if not filtered_df.empty
        else "N/A"
    )

    st.info(
        f"🎭 Top Genre: {top_genre}"
    )

with col3:

    top_rating = (
        filtered_df["rating"]
        .mode()[0]
        if not filtered_df.empty
        else "N/A"
    )

    st.info(
        f"⭐ Top Rating: {top_rating}"
    )

st.markdown("---")

# ==========================================
# CHARTS
# ==========================================

# ==========================================
# CHART 1 - MOVIES VS TV SHOWS
# ==========================================

type_counts = (
    filtered_df["type"]
    .value_counts()
    .reset_index()
)

type_counts.columns = ["Type", "Count"]

pie_fig = px.pie(
    type_counts,
    names="Type",
    values="Count",
    title="Movies vs TV Shows",
    color_discrete_sequence=[
        "#E50914",
        "#831010"
    ],
    template="plotly_dark"
)

# ==========================================
# CHART 2 - RATINGS
# ==========================================

rating_counts = (
    filtered_df["rating"]
    .value_counts()
    .head(10)
    .reset_index()
)

rating_counts.columns = [
    "Rating",
    "Count"
]

rating_fig = px.bar(
    rating_counts,
    x="Rating",
    y="Count",
    title="Top Ratings",
    color_discrete_sequence=["#E50914"],
    template="plotly_dark"
)

# ==========================================
# CHART 3 - COUNTRIES
# ==========================================

country_counts = (
    filtered_df["country"]
    .dropna()
    .value_counts()
    .head(10)
    .reset_index()
)

country_counts.columns = [
    "Country",
    "Count"
]

country_fig = px.bar(
    country_counts,
    x="Count",
    y="Country",
    orientation="h",
    title="Top Countries",
    color_discrete_sequence=["#E50914"],
    template="plotly_dark"
)

# ==========================================
# CHART 4 - GENRES
# ==========================================

genre_counts = (
    filtered_df["main_genre"]
    .value_counts()
    .head(10)
    .reset_index()
)

genre_counts.columns = [
    "Genre",
    "Count"
]

genre_fig = px.bar(
    genre_counts,
    x="Count",
    y="Genre",
    orientation="h",
    title="Top Genres",
    color_discrete_sequence=["#E50914"],
    template="plotly_dark"
)

# ==========================================
# DASHBOARD GRID
# ==========================================

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        rating_fig,
        use_container_width=True
    )

col3, col4 = st.columns(2)

with col3:
    st.plotly_chart(
        country_fig,
        use_container_width=True
    )

with col4:
    st.plotly_chart(
        genre_fig,
        use_container_width=True
    )

# ==========================================
# CONTENT GROWTH
# ==========================================

st.markdown("## 📈 Content Growth Over Time")

year_counts = (
    filtered_df["release_year"]
    .value_counts()
    .sort_index()
    .reset_index()
)

year_counts.columns = [
    "Year",
    "Count"
]

growth_fig = px.line(
    year_counts,
    x="Year",
    y="Count",
    markers=True,
    color_discrete_sequence=["#E50914"],
    template="plotly_dark"
)

growth_fig.update_traces(
    line=dict(width=3)
)

st.plotly_chart(
    growth_fig,
    use_container_width=True
)

# ==========================================
# DATA EXPLORER
# ==========================================

st.markdown("## 📋 Dataset Explorer")

st.dataframe(
    filtered_df[
        [
            "title",
            "type",
            "release_year",
            "rating",
            "country"
        ]
    ],
    use_container_width=True
)

st.download_button(
    "⬇ Download Filtered Data",
    filtered_df.to_csv(index=False),
    file_name="netflix_filtered_data.csv",
    mime="text/csv"
)
