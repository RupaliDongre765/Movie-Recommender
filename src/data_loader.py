import pandas as pd
from pathlib import Path

def load_data(root=Path("data/ml-latest-small")):
    ratings = pd.read_csv(root / "ratings.csv")
    movies = pd.read_csv(root / "movies.csv")
    return ratings, movies
