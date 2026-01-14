import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

print("Loading data...")
# Load the datasets
movies = pd.read_csv('model/tmdb_5000_movies.csv')
credits = pd.read_csv('model/tmdb_5000_credits.csv')

print("Merging datasets...")
# Merge the datasets
movies = movies.merge(credits, on='title')

print("Selecting important columns...")
# Keep only important columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Remove rows with missing values
movies.dropna(inplace=True)

print("Processing genres...")
def convert(text):
    """Convert stringified list to actual list and extract names"""
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'])
    except:
        return []
    return L

def convert_cast(text):
    """Extract top 3 cast members"""
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
            counter += 1
    except:
        return []
    return L

def fetch_director(text):
    """Extract director name from crew"""
    L = []
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
    except:
        return []
    return L

# Apply conversions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

print("Processing overview...")
# Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

print("Removing spaces from names...")
# Remove spaces from names
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

print("Creating tags...")
# Create tags column by combining all features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Select final columns
new_df = movies[['movie_id', 'title', 'tags']]

# Convert tags list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

print("Calculating similarity matrix...")
# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vector)

print("Saving pickle files...")
# Save the processed dataframe and similarity matrix
pickle.dump(new_df, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

print("✓ Preprocessing complete!")
print(f"✓ Processed {len(new_df)} movies")
print("✓ Files saved:")
print("  - model/movie_list.pkl")
print("  - model/similarity.pkl")


