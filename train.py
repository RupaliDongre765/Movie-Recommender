from data.download import download_movielens
from src.data_loader import load_data
from src.recommenders import collaborative_filtering

data_root = download_movielens()
ratings, movies = load_data(data_root)
recs = collaborative_filtering(ratings, user_id=1, top_k=5)
print("Top 5 Recommendations for User 1:")
print(recs)
