import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

data = [
    {"name": "Spice Villa", "cuisine": "Indian", "price_range": "low", "location": "Downtown", "tags": "vegetarian, spicy, family", "rating": 4.2},
    {"name": "Pasta Palace", "cuisine": "Italian", "price_range": "medium", "location": "Uptown", "tags": "pasta, cozy, wine", "rating": 4.5},
    {"name": "Sushi World", "cuisine": "Japanese", "price_range": "high", "location": "Downtown", "tags": "sushi, seafood, elegant", "rating": 4.7},
    {"name": "Burger Barn", "cuisine": "American", "price_range": "low", "location": "Suburb", "tags": "fast-food, burgers, family", "rating": 4.0},
    {"name": "Curry Corner", "cuisine": "Indian", "price_range": "medium", "location": "Downtown", "tags": "curry, spicy, takeaway", "rating": 4.1},
    {"name": "Green Salads", "cuisine": "Healthy", "price_range": "low", "location": "Uptown", "tags": "salad, vegan, fresh", "rating": 4.3},
    {"name": "Steak House", "cuisine": "American", "price_range": "high", "location": "Downtown", "tags": "steak, fine-dining, wine", "rating": 4.6},
    {"name": "Noodle Nook", "cuisine": "Chinese", "price_range": "low", "location": "Suburb", "tags": "noodles, quick, cheap", "rating": 3.9},
    {"name": "Taco Town", "cuisine": "Mexican", "price_range": "medium", "location": "Uptown", "tags": "tacos, spicy, casual", "rating": 4.0},
    {"name": "Paneer Palace", "cuisine": "Indian", "price_range": "high", "location": "Uptown", "tags": "paneer, vegetarian, family", "rating": 4.4},
]

df = pd.DataFrame(data)
def preprocess_restaurants(df):
    df = df.copy()
    for col in ["cuisine", "price_range", "location", "tags"]:
        df[col] = df[col].fillna("").astype(str).str.lower().str.strip()
    df["profile_text"] = df["cuisine"] + " " + df["price_range"] + " " + df["location"] + " " + df["tags"]
    return df
def recommend_restaurants(user_preferences, df, tfidf, tfidf_matrix, top_n=5, min_rating=None, price_constraint=None):
    parts = []
    for key in ["cuisine", "price_range", "location", "tags"]:
        if key in user_preferences and user_preferences[key]:
            parts.append(str(user_preferences[key]).lower())
    user_text = " ".join(parts) if parts else ""
    df_res = df.copy()
 if user_text.strip() == "":
        if min_rating is not None:
            df_res = df_res[df_res["rating"] >= float(min_rating)]
        if price_constraint:
            df_res = df_res[df_res["price_range"] == price_constraint.lower()]
        df_res = df_res.sort_values("rating", ascending=False).head(top_n)
        df_res["score"] = 0.0
        return df_res


    user_vec = tfidf.transform([user_text])
    cosine_similarities = linear_kernel(user_vec, tfidf_matrix).flatten()
    df_res["score"] = cosine_similarities

    if min_rating is not None:
        df_res = df_res[df_res["rating"] >= float(min_rating)]
    if price_constraint:
        df_res = df_res[df_res["price_range"] == price_constraint.lower()]
    df_res = df_res.sort_values(["score", "rating"], ascending=[False, False]).head(top_n)
    return df_res
   samples = [
    {"name": "User A - Likes Indian, low price, vegetarian", "prefs": {"cuisine": "Indian", "price_range": "low", "tags": "vegetarian"}},
    {"name": "User B - Wants sushi or seafood, high price", "prefs": {"cuisine": "Japanese", "tags": "sushi, seafood", "price_range": "high"}},
    {"name": "User C - Healthy vegan, any price", "prefs": {"tags": "vegan, salad"}},
    {"name": "User D - No preference, want top rated", "prefs": {}},
]

results = {}
for s in samples:
    recs = recommend_restaurants(s["prefs"], df, tfidf, tfidf_matrix, top_n=5, min_rating=4.0)
    results[s["name"]] = recs[["name", "cuisine", "price_range", "tags", "rating", "score"]]


for user, table in results.items():
    print("\n--- Recommendations for:", user)
print("\nExample: Recommendations for User A (first 5 rows):")
print(results["User A - Likes Indian, low price, vegetarian"].reset_index(drop=True).head(5))
