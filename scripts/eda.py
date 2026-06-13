import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("../data/netflix_cleaned.csv")

print("=" * 50)
print("EDA STARTED")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# MOVIES VS TV SHOWS
# ==========================================

plt.figure(figsize=(8, 5))
df["type"].value_counts().plot(kind="bar")
plt.title("Movies vs TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../images/content_type_distribution.png")
plt.close()

# ==========================================
# TOP 10 GENRES
# ==========================================

genres = df["listed_in"].str.split(", ").explode()

plt.figure(figsize=(12, 6))
genres.value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Genre")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig("../images/top_10_genres.png")
plt.close()

# ==========================================
# TOP 10 COUNTRIES
# ==========================================

countries = df["country"].str.split(", ").explode()

plt.figure(figsize=(12, 6))
countries.value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Content Producing Countries")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig("../images/top_10_countries.png")
plt.close()

# ==========================================
# CONTENT GROWTH BY YEAR
# ==========================================

plt.figure(figsize=(12, 6))
df["release_year"].value_counts().sort_index().plot()
plt.title("Netflix Content Growth by Release Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig("../images/content_growth_by_year.png")
plt.close()

# ==========================================
# RATINGS DISTRIBUTION
# ==========================================

plt.figure(figsize=(12, 6))
df["rating"].value_counts().head(10).plot(kind="bar")
plt.title("Content Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../images/ratings_distribution.png")
plt.close()

# ==========================================
# MOVIES VS TV SHOWS PIE CHART
# ==========================================

plt.figure(figsize=(7, 7))
df["type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.ylabel("")
plt.title("Movies vs TV Shows Percentage")
plt.tight_layout()
plt.savefig("../images/content_type_pie_chart.png")
plt.close()

# ==========================================
# SUMMARY STATISTICS
# ==========================================

print("\nMovies Count:")
print((df["type"] == "Movie").sum())

print("\nTV Shows Count:")
print((df["type"] == "TV Show").sum())

print("\nTop 5 Countries:")
print(countries.value_counts().head())

print("\nTop 5 Genres:")
print(genres.value_counts().head())

print("\nEDA COMPLETED SUCCESSFULLY")
print("Charts saved inside images folder")

print("=" * 50)