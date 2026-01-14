import pickle
import streamlit as st

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_similarity = []
    
    for i in distances[1:6]:
        recommended_movie_names.append(movies.iloc[i[0]].title)
        # Calculate similarity percentage
        similarity_score = round(i[1] * 100, 2)
        recommended_movie_similarity.append(similarity_score)

    return recommended_movie_names, recommended_movie_similarity


st.header('Movie Recommender System')
movies = pickle.load(open('model/movie_list.pkl','rb'))
similarity = pickle.load(open('model/similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_similarity = recommend(selected_movie)
    
    st.success(f"Top 5 movies similar to '{selected_movie}':")
    
    # Display recommendations in a clean format
    for i, (movie, similarity) in enumerate(zip(recommended_movie_names, recommended_movie_similarity), 1):
        st.markdown(f"### {i}. {movie}")
        st.progress(similarity / 100)
        st.caption(f"Similarity Score: {similarity}%")
        st.divider()





