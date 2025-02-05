# Search-Based Book Recommendation System

This repository contains a **Search-Based Book Recommendation System** that suggests similar books based on their titles using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **cosine similarity**.

## Features
- Cleans and processes book datasets.
- Computes **TF-IDF vectors** for book titles.
- Uses **cosine similarity** to find and recommend similar books.
- Saves precomputed **TF-IDF model** and **TF-IDF matrix** for faster recommendations.

## Installation
Ensure you have **Python 3.x** and install required libraries:
```bash
pip install pandas scikit-learn pickle5
```

## Dataset Preparation
1. Load datasets (`Books.csv` and `Ratings.csv`).
2. Merge them on **ISBN**.
3. Remove missing values and sample **20,000 records**.
4. Save the cleaned dataset as `final_df.csv`.

## Usage
### Load and Train TF-IDF Model
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load dataset
df = pd.read_csv("final_df.csv")

# Drop duplicate book titles
df = df.drop_duplicates(subset=['Book-Title']).reset_index(drop=True)

# Compute TF-IDF matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['Book-Title'])
```

### Book Recommendation Function
```python
def search_based_recommendation(book_title, df, tfidf, tfidf_matrix, top_n=10):
    """Recommend books based on title similarity using TF-IDF and cosine similarity."""
    query_vector = tfidf.transform([book_title])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sim_scores = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)[1:top_n + 1]
    book_indices = [i[0] for i in sim_scores]
    return df[['Book-Title', 'Book-Rating', 'Image-URL-M']].iloc[book_indices].reset_index(drop=True)
```

### Example Usage
```python
book_to_search = "The Fountainhead"
recommended_books = search_based_recommendation(book_to_search, df, tfidf, tfidf_matrix)
print(recommended_books)
```

### Save the Model for Future Use
```python
pickle.dump(tfidf, open("tfidf.pkl", 'wb'))
pickle.dump(tfidf_matrix, open("tfidf_matrix.pkl", 'wb'))
```

## Output
The system returns a list of **top N recommended books** with their **titles, ratings, and image URLs**.
