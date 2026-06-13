import os
import sys
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PATHS
# ==========================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

CSV_PATH  = os.path.join(DATA_DIR,  "netflix_cleaned.csv")

# ==========================================
# LOAD DATA
# ==========================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ==========================================
# FILL MISSING VALUES
# ==========================================

features = [
    "director",
    "cast",
    "listed_in",
    "description"
]

for feature in features:
    df[feature] = df[feature].fillna("")

# ==========================================
# CREATE COMBINED FEATURE
# ==========================================

df["combined_features"] = (
    df["listed_in"]    + " " + df["listed_in"]    + " " +
    df["director"]     + " " +
    df["cast"]         + " " +
    df["description"]
)

print("Combined Features Created")

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = vectorizer.fit_transform(
    df["combined_features"]
)

print("TF-IDF Matrix Created")
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# ==========================================
# COSINE SIMILARITY
# ==========================================

similarity = cosine_similarity(tfidf_matrix)

print("Similarity Matrix Created")
print(f"Similarity Matrix Size: {sys.getsizeof(similarity) / 1e6:.1f} MB")

# ==========================================
# SAVE MODEL FILES
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)

pickle.dump(df, open(os.path.join(MODEL_DIR, "recommender.pkl"), "wb"))
pickle.dump(similarity, open(os.path.join(MODEL_DIR, "similarity.pkl"), "wb"))

print("Model Files Saved Successfully")

print("\nProject Training Completed")
