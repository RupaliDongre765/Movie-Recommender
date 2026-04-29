import requests, zipfile, io, os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_movielens():
    zip_path = DATA_DIR / "ml-latest-small.zip"
    extract_path = DATA_DIR / "ml-latest-small"
    if extract_path.exists():
        print("Dataset already downloaded.")
        return extract_path
    print("Downloading MovieLens dataset...")
    r = requests.get(URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DATA_DIR)
    print("Extraction complete.")
    return extract_path

if __name__ == "__main__":
    download_movielens()
