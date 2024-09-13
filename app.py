import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats

# Import warnings to suppress any warnings
import warnings
warnings.filterwarnings('ignore')

# Define the number of votes needed for a book to be considered popular
target_vote_number = 100
numberOfResult = 5

# Streamlit Title and Introduction
st.title("Book Recommendation System")

# Displaying sections with markdown text
st.markdown("### Book Recommendation Algorithm")
st.markdown("In this application, we recommend books using Collaborative Filtering methods. \
We analyze user preferences and interactions to suggest books you might like based on similar users or items.")

# Reading Data
@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv", nrows=90000)
    books = pd.read_csv("books.csv", nrows=90000)
    books = books[['book_id', 'title', 'authors', 'original_publication_year']].dropna()
    books = books.astype({"title": "string", "authors": "string"})
    return ratings, books

ratings, books = load_data()

# Show dataset information in the sidebar
st.sidebar.header("Data Information")
st.sidebar.write(f"The ratings dataset has {ratings['user_id'].nunique()} unique users.")
st.sidebar.write(f"The ratings dataset has {ratings['book_id'].nunique()} unique books.")
st.sidebar.write(f"The ratings dataset has {ratings['rating'].nunique()} unique ratings.")

# Merge datasets
ratings_books = pd.merge(ratings, books, on='book_id', how='inner')

# User Selection
st.subheader("Select User ID")
userID = st.number_input("Enter a user ID:", min_value=ratings['user_id'].min(), max_value=ratings['user_id'].max(), value=35, step=1)

# Collaborative Filtering Recommendation System
def recommender_system(userID):
    user_book_matrix = ratings_books.pivot_table(index='user_id', columns='book_id', values='rating')
    matrix_norm = user_book_matrix.subtract(user_book_matrix.mean(axis=1), axis='rows')
    user_sim = matrix_norm.T.corr()

    def get_similar_user(userID):
        user_s = user_sim.copy()
        user_similarity_threshold = 0.3
        user_s.drop(index=userID, inplace=True)
        similar_user = user_s[user_s[userID] > user_similarity_threshold][userID].sort_values(ascending=False)
        return similar_user

    # Get similar users
    similar_user = get_similar_user(userID)

    # Narrow down the books
    target_userid_read = matrix_norm[matrix_norm.index == userID].dropna(axis=1, how='all')
    similar_user_books = matrix_norm[matrix_norm.index.isin(similar_user.index)].dropna(axis=1, how='all')

    # Remove books already read by the user
    similar_user_books.drop(target_userid_read.columns, axis=1, inplace=True, errors='ignore')

    # Predict the score for the books
    book_score = {}
    for i in similar_user_books.columns:
        book_rating = similar_user_books[i]
        total = 0
        count = 0
        for u in similar_user.index:
            if pd.isna(book_rating[u]) == False:
                score = similar_user[u] * book_rating[u]
                total += score
                count += 1
        book_score[i] = total / count

    book_score = pd.DataFrame(book_score.items(), columns=['book_id', 'book_score'])
    ranked_book_score = pd.merge(book_score, books, on='book_id', how='inner')

    ranked_book_score = ranked_book_score.sort_values(by='book_score', ascending=False)

    avg_rating = user_book_matrix[user_book_matrix.index == userID].T.mean()[userID]
    ranked_book_score['predicted_rating'] = ranked_book_score['book_score'] + avg_rating

    return ranked_book_score

# Display recommended books for the selected user
st.subheader(f"Recommended Books for User {userID}")
recommendation = recommender_system(userID)
st.dataframe(recommendation[['title', 'authors', 'original_publication_year']].head(10))

# Item-based Collaborative Filtering
st.subheader("Item-Based Collaborative Filtering")

def item_based_recommender_system(userID):
    ratings_books_matrix = ratings_books.pivot_table(index='user_id', columns=['book_id'], values='rating')
    books_ratings_matrix = ratings_books_matrix.transpose()

    books_ratings_user = books_ratings_matrix[userID]
    top_rated_books_id = ratings_books.loc[ratings_books['user_id'] == userID].sort_values(by='rating', ascending=False)['book_id'].tolist()[0]

    ratings = ratings_books_matrix[top_rated_books_id]
    similar_books = ratings_books_matrix.corrwith(ratings)
    similar_books = pd.DataFrame(similar_books, columns=['correlation']).sort_values(by='correlation', ascending=False)

    # Drop the source book and ensure popularity
    similar_books = similar_books[1:]
    rating_votes = pd.DataFrame(ratings_books.groupby('book_id')['rating'].count()).rename(columns={'rating': 'rating_count'})
    similar_books_ratings = similar_books.join(rating_votes['rating_count']).sort_values(by='correlation', ascending=False)
    similar_popular_books = similar_books_ratings.loc[similar_books_ratings['rating_count'] >= target_vote_number].dropna()

    # Trim results from user's already read list
    target_user = ratings_books.loc[ratings_books['user_id'] == userID]
    similar_popular_books = similar_popular_books[~similar_popular_books.index.isin(target_user['book_id'].tolist())]
    most_similar_popular_books_list = similar_popular_books.index.to_list()

    return books.loc[books['book_id'].isin(most_similar_popular_books_list)]

# Show item-based recommended books for the user
item_recommendations = item_based_recommender_system(userID)
st.dataframe(item_recommendations[['title', 'authors', 'original_publication_year']].head(numberOfResult))

# End of Streamlit App
