# No external libraries are allowed to be imported in this file
import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from numpy.linalg import norm
import multiprocessing
import random


"""The u.data dataset contains the ranking assigned by the users 
of a streaming platform to the movies available on the platform."""
# Importing dataset: paste your path to u.data in the following line:
path = "u.data"
df = pd.read_table(path, sep="\t", names=["UserID", "MovieID", "Rating", "Timestamp"])
# print(df.head())  # to check the correct import of the dataset

df = df.pivot_table(index="UserID", columns="MovieID", values="Rating")


# 1. COSINE SIMILARITY:
def similarity_matrix(matrix, k=5, axis=0):
    """
    This function should contain the code to compute the cosine similarity (according to the
    formula seen in the lecture) between users (axis=0) or items (axis=1) and return a dictionary
    where each key represents a user (or item) and the value is a list of the top k most similar
    users or items, along with their similarity scores.

    Args:
        matrix (pd.DataFrame) : user-item rating matrix (df)
        k (int): number of top k similarity rankings to return for each entity (default=5)
        axis (int): 0: calculate similarity scores between users (rows of the matrix),
                    1: calculate similarity scores between items (columns of the matrix)

    Returns:
        similarity_dict (dictionary): dictionary where the keys are users (or items) and
        the values are lists of tuples containing the most similar users (or items) along
        with their similarity scores.

    Note that is NOT allowed to automatically compute cosine similarity using an existing
    function from any package, the computation should follow the formula that has been
    discussed during the lecture and that can be found in the slides.

    Note that it is allowed to convert the DataFrame into a Numpy array for faster computation.
    """
    similarity_dict = {}

    # Convert matrix to numpy array and fill unknowns
    # with zeroes (note: this negatively affects unknowns)
    matrix = matrix.fillna(0).to_numpy()

    # If we are doing item-based, transpose the matrix
    if axis == 1:
        matrix = matrix.T

    # Take the index size of the matrix
    index, _ = matrix.shape

    # Loop over all users / items in the matrix
    for u in range(1, index):
        # Set the value to a temporary variable
        _u = matrix[u][1:]
        # Initialize the similarity dict for this user / item
        similarity_dict[u + 1] = []

        # Loop over all _other_ users / items
        for v in range(1, index):
            # Only if we are not comparing one to itself
            if u != v:
                # Set the value to a temporary variable
                _v = matrix[v][1:]
                # See slide 16/53: Find Nearest Neighbours
                # note that our user_ids and movie_ids are +1 compared to 0-index
                similarity_dict[u + 1] += [
                    (v + 1, np.dot(_u, _v) / (norm(_u) * norm(_v)))
                ]

    # Loop over all users again
    for u in range(1, index):
        # Sort the output array by similarity, descending and slice the
        # first k elements
        similarity_dict[u + 1] = sorted(
            similarity_dict[u + 1], key=lambda x: x[1], reverse=True
        )[:k]

    return similarity_dict


# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement user-based collaborative filtering,
    returning the predicted rate associated to a target user-movie pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        user_similarity (dict): dictionary containing user similarities, obtained using the
                                similarity_matrix function (axis=0)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the computation (default=5)

    Returns:
    predicted_rating (float): predicted rating according to user-based collaborative filtering

    Note that the selected value of k in this functions must be less or equal to the value of k
    selected in the similarity_matrix function used to create the user_similarity matrix provided
    as input to obtain a correct result.
    """
    # Slice the top k elements and convert to numpy array
    top_k = user_similarity[user_id][:k]
    _user_item_matrix = user_item_matrix.to_numpy()

    # nansum (nan = 0) of the best-k-user's rating of the movie times the similarity
    numerator = np.nansum(
        [_user_item_matrix[i[0] - 1][movie_id - 1] * i[1] for i in top_k]
    )
    # take the similarities of the top-k users
    denominator = np.nansum([i[1] for i in top_k])

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement item-based collaborative filtering,
    returning the predicted rate associated to a target user-movie pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        item_similarity (dict): dictonary containing item similarities, obtained using the
                                similarity_matrix function (axis=1)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the computation (default=5)

    Returns:
    predicted_rating (float): predicted rating according to item-based collaborative filtering

    Note that the selected value of k in this functions must be less or equal to the value of k
    selected in the similarity_matrix function used to create the item_similarity matrix provided
    as input to obtain a correct result.
    """
    # Slice the top k elements and convert the matrix to a numpy array
    # and transpose as we are looking based on items
    top_k = item_similarity[movie_id][:k]
    _user_item_matrix = user_item_matrix.to_numpy().T

    # Take the most similar movies and multiply with the ratings
    numerator = np.nansum(
        [_user_item_matrix[i[0] - 1][user_id - 1] * i[1] for i in top_k]
    )
    # take the similarities of the top-k movies
    denominator = np.nansum([i[1] for i in top_k])

    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


# 3. MATRIX FACTORISATION - UV DECOMPOSITION - ALTERNATING LEAST SQUARES (ALS):
"""This class performs matrix factorization using Alternating Least Squares (ALS) to decompose the user-item 
   rating matrix into user (U) and item (V) feature matrices. The goal is to predict user ratings for items.
   Your task is just to understand the provided implementation of this more complex algorithm and to complete 
   the update_V function, which updates the item matrix V. To do so, refer to the update_U function as it 
   follows the same logic for updating the user matrix U."""


class UVDecomposition:
    def __init__(
        self,
        path,
        num_factors,
        num_iters,
        seed,
        save=False,
        feature_matrices_path="",
        random=False,
        data=None,
    ):
        """
        num_factors: implicit feature
        num_iters: number of iteration

        """
        self.num_factors = num_factors
        self.num_iters = num_iters
        self.seed = seed
        self.save = save
        self.feature_matrices_path = feature_matrices_path

        # Read the data
        if data is not None:
            self.data = pd.DataFrame(data, columns=["UserID", "MovieID", "Rating"])
        else:
            # Read data from file if no data is provided
            self.data = pd.read_table(
                path, sep="\t", names=["UserID", "MovieID", "Rating", "Timestamp"]
            )

        self.user_item_matrix = self.data.pivot_table(
            index="UserID", columns="MovieID", values="Rating"
        ).fillna(0)
        self.R = self.user_item_matrix.to_numpy()

        np.random.seed(self.seed)

        # Initialize the user feature matrix U and the item feature matrix V
        num_users, num_items = self.R.shape
        self.U = np.random.normal(
            scale=1.0 / self.num_factors, size=(num_users, self.num_factors)
        )
        self.V = np.random.normal(
            scale=1.0 / self.num_factors, size=(num_items, self.num_factors)
        )

    def update_U(self, index, data_train, U, V):
        """
        r: the user's/movie's index
        c: the index of the feature to be updated
        """
        user_index, feature_update = index
        M = []

        # Get rating data related to user R
        for row in data_train:
            if row[0] == (user_index + 1):
                M.append(row)

        M = np.array(M)
        sum_1, sum_2 = 0, 0

        for row in range(M.shape[0]):
            m = int(M[row, 1]) - 1
            pred = (
                np.dot(U[user_index, :], V[:, m])
                - U[user_index, feature_update] * V[feature_update, m]
            )
            sum_1 += V[feature_update, m] * (M[row, 2] - pred)
            sum_2 += V[feature_update, m] ** 2

        if sum_2 == 0:
            sum_2 = 0.001  # Prevent the denominator from being 0

        U[user_index, feature_update] = sum_1 / sum_2
        return U

    def update_V(self, index, data_train, U, V):
        """
        This function should contain the code to update the item feature matrix V, according to
        the Alternating Least Squares (ALS) algorithm.

        Args:
            index (tuple): Tuple representing the (feature_update, movie_index) pair for V.
            data_train (np.array): Array containing user-item rating data.
            U (np.array): The user feature matrix.
            V (np.array): The item feature matrix.

        Returns:
            Updated item feature matrix V.

        Note that is allowed and recommended to look at the implementation of the update_U
        function, that implements the same type of update on matrix U.
        """
        feature_update, movie_index = index
        M = []

        # Get rating data related to movie R
        for movie in data_train:
            if movie[1] == (movie_index + 1):
                M.append(movie)

        M = np.array(M)
        sum_1, sum_2 = 0, 0

        for rating in range(M.shape[0]):
            user_index = int(M[rating, 0]) - 1
            pred = (
                np.dot(U[user_index, :], V[:, movie_index])
                - U[user_index, feature_update] * V[feature_update, movie_index]
            )
            sum_1 += U[user_index, feature_update] * (M[rating, 2] - pred)
            sum_2 += U[user_index, feature_update] ** 2

        if sum_2 == 0:
            sum_2 = 0.001  # Prevent the denominator from being 0

        V[feature_update, movie_index] = sum_1 / sum_2
        return V

    def _train_iteration(self, U, V, data_train):
        """
        Updated user and item matrix
        """
        u_index = list(np.ndindex(U.shape))
        v_index = list(np.ndindex(V.shape))

        random.shuffle(u_index)
        random.shuffle(v_index)

        while len(u_index) > 0 or len(v_index) > 0:
            if len(u_index) > 0:
                u = u_index.pop()
                U = self.update_U(u, data_train, U, V)

            if len(v_index) > 0:
                v = v_index.pop()
                V = self.update_V(v, data_train, U, V)

        return U, V

    def train(self):
        """
        train the model
        """
        data_train = self.data[["UserID", "MovieID", "Rating"]].to_numpy()
        for i in range(self.num_iters):
            print(f"Iteration {i+1}/{self.num_iters}")
            self.U, self.V = self._train_iteration(self.U, self.V.T, data_train)
            self.V = self.V.T

        if self.save:
            np.savez(self.feature_matrices_path, U=self.U, V=self.V)

    def predict(self, user, item):
        user_index = user - 1
        item_index = item - 1

        prediction = np.dot(self.U[user_index, :], self.V[item_index, :].T)
        return prediction


if __name__ == "__main__":
    # This main section is intended for testing your implemented functions:

    # You can use this section for testing the similarity_matrix function:
    # Return the top 5 most similar users to user 3:
    # user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    # print(user_similarity_matrix.get(3, []))

    # Return the top 5 most similar items to item 10:
    # item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    # print(item_similarity_matrix.get(10, []))

    # You can use this section for testing the user_based_cf and the item_based_cf functions:
    # Return the predicted ratings assigned by user 13 to movie 100:
    user_id = 13
    movie_id = 100

    # u_predicted_rating = user_based_cf(
    #     user_id, movie_id, user_similarity_matrix, user_item_matrix=df, k=5
    # )
    # print(
    #     f"predicted user {user_id} rating for movie {movie_id}, according to user-based collaborative filtering is: {u_predicted_rating:.2f}"
    # )

    # i_predicted_rating = item_based_cf(
    #     user_id, movie_id, item_similarity_matrix, user_item_matrix=df, k=5
    # )
    # print(
    #     f"predicted user {user_id} rating for movie {movie_id}, according to item-based collaborative filtering is: {i_predicted_rating:.2f}"
    # )

    # You can use this section for testing the update_V function within the UVDecomposition class:
    num_factors = 10
    num_iters = 5
    seed = 42
    save = True

    uv_model = UVDecomposition(path, num_factors, num_iters, seed, save)
    uv_model.train()

    # Return the predicted ratings assigned by user 13 to movie 100:

    user_id = 13
    movie_id = 100
    predicted_rating = uv_model.predict(user_id, movie_id)
    print(
        f"predicting User {user_id} rating for movie {movie_id}, according to the UV-decomposition method is: {predicted_rating:.2f}"
    )
