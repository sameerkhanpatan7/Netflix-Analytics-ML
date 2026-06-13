import pandas as pd

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("../data/netflix.csv")

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

# ==========================================
# BASIC INFORMATION
# ==========================================

print("\nShape of Dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ==========================================
# MISSING VALUES BEFORE CLEANING
# ==========================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ==========================================
# REMOVE DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()

print(f"\nDuplicate Rows Found: {duplicates}")

df.drop_duplicates(inplace=True)

print(f"Rows after removing duplicates: {df.shape[0]}")

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Not Rated")
df["duration"] = df["duration"].fillna("Unknown")

# Remove rows with missing title
df.dropna(subset=["title"], inplace=True)

# ==========================================
# DATE CONVERSION
# ==========================================

df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)

# ==========================================
# CLEAN STRING COLUMNS
# ==========================================

string_cols = [
    "title",
    "director",
    "cast",
    "country",
    "rating",
    "listed_in",
    "description"
]

for col in string_cols:
    df[col] = df[col].astype(str).str.strip()

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month

# Fill newly created columns
df["year_added"] = df["year_added"].fillna(0).astype(int)
df["month_added"] = df["month_added"].fillna(0).astype(int)

# ==========================================
# FINAL DATASET INFORMATION
# ==========================================

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

# ==========================================
# SAVE CLEANED DATASET
# ==========================================

output_path = "../data/netflix_cleaned.csv"

df.to_csv(output_path, index=False)

# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("\n" + "=" * 50)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print(f"Cleaned Dataset Saved To: {output_path}")
print("=" * 50)