import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("F:/AI DATASET/ml-latest-small/ml-latest-small/movies.csv")
movies['genres'] = movies['genres'].fillna('')

# Convert movie titles and genres to lowercase for uniformity
movies['title'] = movies['title'].str.lower().str.strip()

# Compute TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to find closest matching movie title
def find_closest_match(movie_name):
    movie_name = movie_name.lower().strip()
    best_match, score = process.extractOne(movie_name, movies['title'].tolist())

    if score < 85:  # Adjust threshold for better accuracy
        return None
    return best_match

# Movie recommendation function
def get_recommendations(movie_title):
    match = find_closest_match(movie_title)

    if not match:
        return None  # Return None if no match found

    idx = movies[movies['title'] == match].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations

    recommended_movies = [movies['title'].iloc[i[0]].title() for i in sim_scores]  # Capitalize movie titles
    return recommended_movies

# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route for getting recommendations
@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form["movie_name"]
    recommendations = get_recommendations(movie_name)

    if recommendations is None:
        return jsonify({"error": "Movie not found"}), 404

    return jsonify({"movies": recommendations})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
