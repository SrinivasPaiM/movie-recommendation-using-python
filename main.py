import pandas as pd
import numpy as np

# Load datasets
ratings_data = pd.read_csv('data/ratings.csv')
movies_data = pd.read_csv('data/movies.csv')

# Pivot
user_movie_matrix = ratings_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Compute cosine similarity between two vectors
def compute_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Generate a similarity matrix for all users
def build_similarity_matrix(matrix):
    total_users = matrix.shape[0]
    similarity_matrix = np.zeros((total_users, total_users))

    for user1 in range(total_users):
        for user2 in range(total_users):
            if user1 != user2:
                similarity_matrix[user1][user2] = compute_cosine_similarity(matrix[user1], matrix[user2])

    return similarity_matrix

# Calculate user similarities
user_similarity_matrix = build_similarity_matrix(user_movie_matrix.values)

# Generate movie recommendations
def get_movie_recommendations(user_id, num_recommendations):
    user_idx = user_id - 1
    similar_users_indices = np.argsort(-user_similarity_matrix[user_idx])[:num_recommendations + 1]

    movie_recommendations = {}
    for sim_user_idx in similar_users_indices:
        if sim_user_idx != user_idx:
            sim_user_ratings = user_movie_matrix.iloc[sim_user_idx]
            for movie_id, rating in sim_user_ratings.items():
                if rating > 0 and user_movie_matrix.iloc[user_idx][movie_id] == 0:
                    if movie_id not in movie_recommendations:
                        movie_recommendations[movie_id] = 0
                    movie_recommendations[movie_id] += rating

    sorted_movie_ids = sorted(movie_recommendations, key=movie_recommendations.get, reverse=True)[:num_recommendations]
    recommended_movies_df = movies_data[movies_data['movieId'].isin(sorted_movie_ids)]

    return recommended_movies_df

if __name__ == '__main__':
    # Request user input
    target_user_id = int(input("Please enter your user ID: "))
    number_of_recommendations = int(input("Enter the number of recommendations you would like to receive: "))

    # Display the movie recommendations
    recommended_movies = get_movie_recommendations(target_user_id, number_of_recommendations)
    if recommended_movies.empty:
        print("No recommendations could be found.")
    else:
        print("Recommended Movies:")
        for idx, movie in recommended_movies.iterrows():
            print(movie['title'])
