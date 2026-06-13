import os
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "model"
)

CSV_PATH = os.path.join(
    DATA_DIR,
    "netflix_cleaned.csv"
)

# ==========================================
# LOAD DATA
# ==========================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("✅ Dataset Loaded")
print("Shape:", df.shape)

# ==========================================
# HANDLE MISSING VALUES
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
# CREATE COMBINED FEATURES
# ==========================================

df["combined_features"] = (
    df["listed_in"] + " " +
    df["listed_in"] + " " +
    df["director"] + " " +
    df["cast"] + " " +
    df["description"]
)

print("✅ Combined Features Created")

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

print("✅ TF-IDF Matrix Created")
print("Shape:", tfidf_matrix.shape)

# ==========================================
# CREATE MODEL FOLDER
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# ==========================================
# SAVE FILES
# ==========================================

pickle.dump(
    df,
    open(
        os.path.join(
            MODEL_DIR,
            "recommender.pkl"
        ),
        "wb"
    )
)

pickle.dump(
    vectorizer,
    open(
        os.path.join(
            MODEL_DIR,
            "vectorizer.pkl"
        ),
        "wb"
    )
)

pickle.dump(
    tfidf_matrix,
    open(
        os.path.join(
            MODEL_DIR,
            "tfidf_matrix.pkl"
        ),
        "wb"
    )
)

print("✅ recommender.pkl saved")
print("✅ vectorizer.pkl saved")
print("✅ tfidf_matrix.pkl saved")

print("\n🎉 Training Completed Successfully")