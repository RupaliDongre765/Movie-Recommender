import streamlit as st
from data.download import download_movielens
from src.data_loader import load_data
from src.recommenders import popularity_recommender, content_based_recommender, collaborative_filtering

st.title("🎬 Intermediate Movie Recommender (Python 3.14 Compatible)")

data_root = download_movielens()
ratings, movies = load_data(data_root)

option = st.selectbox("Choose recommender type", ["Popularity", "Content-Based", "Collaborative"])
if option == "Popularity":
    k = st.slider("Top K", 5, 20, 10)
    recs = popularity_recommender(ratings, movies, top_k=k)
    st.dataframe(recs)
elif option == "Content-Based":
    movie = st.selectbox("Select a movie", movies["title"].head(1000))
    recs = content_based_recommender(movie, movies, top_k=10)
    st.dataframe(recs)
else:
    user = st.number_input("User ID", min_value=1, value=1, step=1)
    recs = collaborative_filtering(ratings, user, top_k=10)
    st.dataframe(recs)
