import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def popularity_recommender(ratings, movies, top_k=10):
    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    count_ratings = ratings.groupby("movieId")["rating"].count()
    df = pd.DataFrame({
        "avg_rating": avg_ratings,
        "count": count_ratings
    }).merge(movies, on="movieId")
    df = df[df["count"] > 20]
    df = df.sort_values(["avg_rating", "count"], ascending=False)
    return df.head(top_k)[["title", "genres", "avg_rating", "count"]]

def content_based_recommender(movie_title, movies, top_k=10):
    cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = cv.fit_transform(movies["genres"])
    cosine_sim = cosine_similarity(genre_matrix)
    indices = pd.Series(movies.index, index=movies["title"])
    if movie_title not in indices:
        return pd.DataFrame()
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["title", "genres"]].assign(similarity=[s for _, s in sim_scores])

def collaborative_filtering(ratings, user_id, top_k=10):
    pivot = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    user_means = pivot.mean(axis=1)
    centered = pivot.sub(user_means, axis=0).fillna(0)
    sim = cosine_similarity(centered)
    np.fill_diagonal(sim, 0)
    user_idx = user_id - 1 if user_id - 1 < sim.shape[0] else 0
    sim_scores = sim[user_idx]
    similar_users = np.argsort(sim_scores)[::-1][:10]
    weighted_ratings = np.dot(sim_scores[similar_users], centered.iloc[similar_users])
    pred_scores = weighted_ratings / (np.abs(sim_scores[similar_users]).sum() + 1e-9)
    recs = pd.DataFrame({
        'movieId': pivot.columns,
        'predicted_rating': pred_scores
    }).sort_values('predicted_rating', ascending=False)
    return recs.head(top_k)
