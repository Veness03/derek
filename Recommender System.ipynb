{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "55a56040",
   "metadata": {
    "id": "55a56040",
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "# Book Recommendation Algorithm\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "39d76715-1c07-4564-b263-a21286c17bb7",
   "metadata": {
    "id": "5d191514"
   },
   "outputs": [],
   "source": [
    "# Import Python Libraries (Lau Chien Yi & Ooi Jin Kun)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "779b7900",
   "metadata": {
    "id": "779b7900"
   },
   "source": [
    "In this step, will be import Python libraries: 'pandas', 'numpy' and 'scipy.stats'. These library are used for data processing and calculations.\n",
    "\n",
    "Other than that, also need to import 'seaborn' for visualization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa022b02",
   "metadata": {
    "id": "fa022b02"
   },
   "outputs": [],
   "source": [
    "# Data processing\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import scipy.stats\n",
    "\n",
    "# Visualization\n",
    "import seaborn as sb\n",
    "\n",
    "#define variable\n",
    "target_vote_number = 100\n",
    "numberOfResult = 5\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72db7a7a",
   "metadata": {
    "id": "72db7a7a"
   },
   "source": [
    "# Download and Read Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ab0ed19",
   "metadata": {
    "id": "0ab0ed19"
   },
   "source": [
    "### Rating Dataset :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f68af37",
   "metadata": {
    "id": "3f68af37"
   },
   "outputs": [],
   "source": [
    "ratings = pd.read_csv(\"ratings.csv\",nrows=90000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae21fd82",
   "metadata": {
    "id": "ae21fd82",
    "outputId": "419dfcbc-e8d1-44e4-caed-e235fa359fd4"
   },
   "outputs": [],
   "source": [
    "ratings.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc319bc3",
   "metadata": {
    "id": "cc319bc3",
    "outputId": "95ecef7c-1287-47f8-bff3-871af50ba066"
   },
   "outputs": [],
   "source": [
    "# Get ratings dataset information\n",
    "ratings.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f75ca2d2",
   "metadata": {
    "id": "f75ca2d2",
    "outputId": "0ea28853-9aec-4d15-ca98-c7af68be3272"
   },
   "outputs": [],
   "source": [
    "# Number of users\n",
    "print('The ratings dataset has', ratings['user_id'].nunique(), 'unique users')\n",
    "\n",
    "# Number of books\n",
    "print('The ratings dataset has', ratings['book_id'].nunique(), 'unique books')\n",
    "\n",
    "# Number of ratings\n",
    "print('The ratings dataset has', ratings['rating'].nunique(), 'unique ratings')\n",
    "\n",
    "# List of unique ratings\n",
    "print('The unique ratings are', sorted(ratings['rating'].unique()))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "56d9be05",
   "metadata": {
    "id": "56d9be05"
   },
   "source": [
    "### Book Dataset :\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a70af8a3",
   "metadata": {
    "id": "a70af8a3"
   },
   "outputs": [],
   "source": [
    "books = pd.read_csv(\"books.csv\",nrows=90000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7926f074",
   "metadata": {
    "id": "7926f074",
    "outputId": "14d36971-937c-4792-c155-54bb7edcf749"
   },
   "outputs": [],
   "source": [
    "books.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02c3c4a4",
   "metadata": {
    "id": "02c3c4a4",
    "outputId": "35112725-3720-4402-e9d6-14685e217a64"
   },
   "outputs": [],
   "source": [
    "books.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f9ab649",
   "metadata": {
    "id": "7f9ab649",
    "outputId": "60c3efbf-c2c7-451e-fe48-ab22a78c144c",
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Filter data\n",
    "books = books[['book_id','title','authors','original_publication_year']]\n",
    "\n",
    "books.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c58b4844",
   "metadata": {
    "id": "c58b4844",
    "outputId": "bf7418a8-d056-4a18-a4f1-0c97949daddf"
   },
   "outputs": [],
   "source": [
    "# Get books dataset information\n",
    "books.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aeec8864",
   "metadata": {
    "id": "aeec8864"
   },
   "source": [
    "# Data Preprosessing\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9d76084",
   "metadata": {
    "id": "a9d76084"
   },
   "source": [
    "### Change Variable type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79af72b5",
   "metadata": {
    "id": "79af72b5",
    "outputId": "7d509379-5dd9-4962-fd8c-83cb94d27999"
   },
   "outputs": [],
   "source": [
    "# Change object data type to string data type using astype()\n",
    "books = books.astype({\"title\":\"string\",\"authors\":\"string\"})\n",
    "books.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09fca602-5be0-411d-a05a-2cdd7773a053",
   "metadata": {
    "id": "735eeda1"
   },
   "outputs": [],
   "source": [
    "### Check Missing Value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4efd108b",
   "metadata": {
    "id": "4efd108b",
    "outputId": "cc5348a3-9c18-4142-f5dd-a4311823c54e"
   },
   "outputs": [],
   "source": [
    "ratings.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5d897ae",
   "metadata": {
    "id": "b5d897ae",
    "outputId": "33592af2-f850-4c73-b9d5-8d4bd7617f52"
   },
   "outputs": [],
   "source": [
    "books.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e29a6115",
   "metadata": {
    "id": "e29a6115",
    "outputId": "009094c9-1ce8-4822-b42b-34f4672ceb84"
   },
   "outputs": [],
   "source": [
    "# Drop missing values in original_publication_year column in books dataset\n",
    "books.dropna(axis=0, inplace = True)\n",
    "books.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8445a118",
   "metadata": {
    "id": "8445a118"
   },
   "source": [
    "### Merge Dataset by book_id\n",
    "\n",
    "Using 'book_id' as the matching key, then append book information to the ratings dataset and named it as 'ratings_book'. Then now we can have the book title and book ratings in the same dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "210a7dae",
   "metadata": {
    "id": "210a7dae",
    "outputId": "a4b76050-c53e-4e50-ee83-8536a37e2937"
   },
   "outputs": [],
   "source": [
    "# Merge ratings and books datasets\n",
    "ratings_books = pd.merge(ratings, books, on='book_id', how='inner')\n",
    "\n",
    "ratings_books.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "32c20d79",
   "metadata": {
    "id": "32c20d79",
    "outputId": "dce3cba9-033e-40b1-86b7-3aeaa306043b"
   },
   "outputs": [],
   "source": [
    "ratings_books.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f29b4ab5",
   "metadata": {
    "id": "f29b4ab5",
    "outputId": "2f55c7bd-d1ee-4e50-8b85-b6f97f7cbba6"
   },
   "outputs": [],
   "source": [
    "# Number of users\n",
    "print('The ratings dataset has', ratings_books['user_id'].nunique(), 'unique users')\n",
    "# Number of books\n",
    "print('The ratings dataset has', ratings_books['book_id'].nunique(), 'unique books')\n",
    "# Number of ratings\n",
    "print('The ratings dataset has', ratings_books['rating'].nunique(), 'unique ratings')\n",
    "# List of unique ratings\n",
    "print('The unique ratings are', sorted(ratings_books['rating'].unique()))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "569feb59",
   "metadata": {
    "id": "569feb59"
   },
   "source": [
    "# User-Based Collaborative Filtering (Lau Chien Yi)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "577cf87e",
   "metadata": {
    "id": "577cf87e"
   },
   "source": [
    "Defination :\n",
    "\n",
    "User-Based Collaborative Filtering makes recommendations based on user product\n",
    "interactions in the past. The assumption behind the algorithm is that similar\n",
    "users like similar products.\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bad1da19",
   "metadata": {
    "id": "bad1da19"
   },
   "source": [
    "### Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c5fe9b9",
   "metadata": {
    "id": "7c5fe9b9",
    "outputId": "c26fca55-a15d-49c5-e274-527c79cf433a"
   },
   "outputs": [],
   "source": [
    "# Aggregate by books\n",
    "agg_ratings_books = ratings_books.groupby('book_id').agg(mean_rating = ('rating', 'mean'),number_of_ratings = ('rating', 'count')).reset_index()\n",
    "\n",
    "agg_ratings_books.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08abfe8e",
   "metadata": {
    "id": "08abfe8e",
    "outputId": "ae90ef21-b803-4edb-e92d-53947fb11b1b"
   },
   "outputs": [],
   "source": [
    "# Check popular based on the number of ratings they have received\n",
    "agg_ratings_books.sort_values(by='number_of_ratings', ascending=False).head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "31bd518f",
   "metadata": {
    "id": "31bd518f",
    "outputId": "a4f97292-d223-4853-a4a1-517123479eb8"
   },
   "outputs": [],
   "source": [
    "# Visualization\n",
    "sb.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_books)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0645187e",
   "metadata": {
    "id": "0645187e"
   },
   "source": [
    "### Create User-Book Matrix (pivot table)¶\n",
    "Transform the dataset into a matrix format. The rows of the matrix are users, and the columns of the matrix are books. The value of the matrix is the user rating of the books if there is a rating. Otherwise, it shows ‘NaN’."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aab9d0f5",
   "metadata": {
    "id": "aab9d0f5",
    "outputId": "b1d9cbdd-b557-4a41-8794-c26059bcbba7"
   },
   "outputs": [],
   "source": [
    "# Create user-book matrix\n",
    "user_book_matrix = ratings_books.pivot_table(index='user_id', columns='book_id', values='rating')\n",
    "user_book_matrix.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb80f602",
   "metadata": {
    "id": "cb80f602"
   },
   "source": [
    "### Data Normalization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e2333ab6",
   "metadata": {
    "id": "e2333ab6",
    "outputId": "b11b1d07-133d-42db-95fd-3ef4323260c7"
   },
   "outputs": [],
   "source": [
    "# Normalize user-book matrix\n",
    "matrix_norm = user_book_matrix.subtract(user_book_matrix.mean(axis=1), axis = 'rows')\n",
    "# This helps in centering the ratings around each user's average, allowing you to identify whether a user rated a particular book higher or lower than their average.\n",
    "\n",
    "matrix_norm.head()\n",
    "# negative value = books with a rating less than the user's average rating\n",
    "# positive value = books with a rating more than the user's average rating"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a03d975",
   "metadata": {
    "id": "0a03d975"
   },
   "source": [
    "### Pearson Correlation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1ca06ffd",
   "metadata": {
    "id": "1ca06ffd",
    "outputId": "f503b694-9144-4c0e-984e-92f2097c9b1d"
   },
   "outputs": [],
   "source": [
    "# User similarity matrix using Pearson correlation\n",
    "user_sim = matrix_norm.T.corr()\n",
    "\n",
    "user_sim.head()\n",
    "# positive value = similar user (same book preference)\n",
    "# neagative value = not similar user (opposite book preference)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "911c092d",
   "metadata": {
    "id": "911c092d"
   },
   "source": [
    "### Identify Similar User (Given Scenario)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5588a4e0",
   "metadata": {
    "id": "5588a4e0",
    "outputId": "9534abf9-ba46-4dcb-9a9a-6b87d7cf0133"
   },
   "outputs": [],
   "source": [
    "# Make a copy\n",
    "user_similarity = user_sim.copy()\n",
    "\n",
    "# Pick a target user\n",
    "target_userID = 35\n",
    "\n",
    "# Remove target user ID from the user_similarity matrix\n",
    "user_similarity.drop(index=target_userID, inplace=True)\n",
    "\n",
    "user_similarity.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7816343f",
   "metadata": {
    "id": "7816343f",
    "outputId": "3892817c-7dd0-416c-95d7-6dca12291791"
   },
   "outputs": [],
   "source": [
    "# Number of similar users to get (Top 10 most similar user for target user)\n",
    "n = 10\n",
    "\n",
    "# User similarity threshold (to make sure the Pearson correlation >0.3)\n",
    "user_similarity_threshold = 0.3\n",
    "\n",
    "# Get top n similar users\n",
    "top_similar_users = user_similarity[user_similarity[target_userID]>user_similarity_threshold][target_userID].sort_values(ascending=False)[:n]\n",
    "\n",
    "print(f'Top {n} similar users for user {target_userID} :\\n', top_similar_users)\n",
    "\n",
    "# this code calculates and prints the top similar users for the specified\n",
    "# target user, considering the user similarity threshold. These similar users\n",
    "# can be used in collaborative filtering recommendation systems to suggest\n",
    "# books that users with similar preferences enjoyed."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f74dcb59",
   "metadata": {
    "id": "f74dcb59"
   },
   "source": [
    "### Input userID to identify the similar user that have read the books"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11bb989d",
   "metadata": {
    "id": "11bb989d"
   },
   "outputs": [],
   "source": [
    "def get_similar_user(userID):\n",
    "    user_s = user_sim.copy()\n",
    "    user_similarity_threshold = 0.3\n",
    "\n",
    "    user_s.drop(index=userID, inplace =True)\n",
    "    similar_user = user_s[user_s[userID] > user_similarity_threshold][userID].sort_values(ascending = False)\n",
    "    return similar_user"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c9ff39f",
   "metadata": {
    "id": "9c9ff39f",
    "outputId": "a51998c7-c976-461d-f0fc-bb4b10f4a9c9"
   },
   "outputs": [],
   "source": [
    "userID = int(input(\"Enter userID: \"))\n",
    "\n",
    "while userID not in ratings['user_id'].unique():\n",
    "    userID = int(input(\"Enter valid userID: \"))\n",
    "\n",
    "sim_users = get_similar_user(userID)\n",
    "print(f'\\nSimilar users for user {userID} :\\n', sim_users)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c1d5841b",
   "metadata": {
    "id": "c1d5841b"
   },
   "source": [
    "### Books that have been read by target user"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97b0b082",
   "metadata": {
    "id": "97b0b082",
    "outputId": "ac2e03d8-afe6-4b13-d06a-6ae747b18d01"
   },
   "outputs": [],
   "source": [
    "target_userid_read = matrix_norm[matrix_norm.index == target_userID].dropna(axis=1, how='all')\n",
    "target_userid_read.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23d70dda",
   "metadata": {
    "id": "23d70dda"
   },
   "source": [
    "### Books that similar user read"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4168435b",
   "metadata": {
    "id": "4168435b",
    "outputId": "310ddd4f-3aaf-4d4a-e7b9-90f297eb9388"
   },
   "outputs": [],
   "source": [
    "similar_user_books = matrix_norm[matrix_norm.index.isin(top_similar_users.index)].dropna(axis=1, how='all')\n",
    "similar_user_books.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30727682",
   "metadata": {
    "id": "30727682"
   },
   "source": [
    "### Remove the books that have been read by target user from the similar_user_books"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f8c49f1",
   "metadata": {
    "id": "1f8c49f1",
    "outputId": "a7be477d-d457-4edc-99cf-d7d5b5b61833"
   },
   "outputs": [],
   "source": [
    "similar_user_books.drop(target_userid_read.columns,axis=1, inplace=True, errors='ignore')\n",
    "\n",
    "similar_user_books.head()\n",
    "#5 books will be removed (according to the target_userid_read list)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28330c91",
   "metadata": {
    "id": "28330c91"
   },
   "source": [
    "### User Based Recommended Result\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "492e4c69",
   "metadata": {
    "id": "492e4c69",
    "outputId": "5ac9b0f3-4276-4ec2-fca7-89217fe310bd"
   },
   "outputs": [],
   "source": [
    "book_score = {}\n",
    "\n",
    "for i in similar_user_books.columns:\n",
    "\n",
    "  # Get the ratings for book i\n",
    "  book_rating = similar_user_books[i]\n",
    "\n",
    "  # Store the total score\n",
    "  total = 0\n",
    "\n",
    "  # Number of scores\n",
    "  count = 0\n",
    "\n",
    "\n",
    "  for u in top_similar_users.index:\n",
    "    # If the book has rating\n",
    "    if pd.isna(book_rating[u]) == False:\n",
    "      # Score = sum of user similarity score * book rating\n",
    "      score = top_similar_users[u] * book_rating[u]\n",
    "      # Total up the score\n",
    "      total += score\n",
    "      # Update number of scores\n",
    "      count +=1\n",
    "  # Calculate average score for the book\n",
    "  book_score[i] = total / count\n",
    "\n",
    "book_score = pd.DataFrame(book_score.items(), columns=['book_id', 'book_score'])\n",
    "\n",
    "ranked_book_score = pd.merge(book_score, books, on = 'book_id', how='inner')\n",
    "\n",
    "# Sort the books by score\n",
    "ranked_book_score = ranked_book_score.sort_values(by='book_score', ascending=False)\n",
    "\n",
    "\n",
    "\n",
    "# m = number of book recommendation\n",
    "m = 10\n",
    "ranked_book_score.head(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "880ee5ef",
   "metadata": {
    "id": "880ee5ef",
    "outputId": "8080d94c-aae5-450a-ed44-8ee47d2264fa"
   },
   "outputs": [],
   "source": [
    "# Average rating for the target user\n",
    "avg_rating = user_book_matrix[user_book_matrix.index == target_userID].T.mean()[target_userID]\n",
    "\n",
    "print(f'The average book rating for user {target_userID} is {avg_rating:.2f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2f7737a",
   "metadata": {
    "id": "d2f7737a",
    "outputId": "78858935-a4aa-4cef-9f8d-a8485b00ace2"
   },
   "outputs": [],
   "source": [
    "# Calcuate the predicted rating\n",
    "ranked_book_score['predicted_rating'] = ranked_book_score['book_score'] + avg_rating\n",
    "\n",
    "ranked_book_score.head(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1cdc8a8d",
   "metadata": {
    "id": "1cdc8a8d",
    "outputId": "e118c134-c4d4-45cb-d238-e1b39502ef5e"
   },
   "outputs": [],
   "source": [
    "## Book recommendation\n",
    "ranked_book_score = ranked_book_score.drop(['book_score','predicted_rating'], axis=1)\n",
    "ranked_book_score.rename(columns = {'book_id':'Book ID','title':'Title','authors':'Author','original_publication_year':'Publish Year'}, inplace=True)\n",
    "print(f'Top {m} book recommendations for user {target_userID}:')\n",
    "ranked_book_score.head(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1edd5f8",
   "metadata": {
    "id": "c1edd5f8"
   },
   "outputs": [],
   "source": [
    "def recommender_system(userID):\n",
    "    # Find similar user\n",
    "\n",
    "    similar_user = get_similar_user(userID)\n",
    "\n",
    "    # Narrow down the book\n",
    "    target_userid_read = matrix_norm[matrix_norm.index == userID].dropna(axis=1, how='all')\n",
    "    similar_user_books = matrix_norm[matrix_norm.index.isin(similar_user.index)].dropna(axis=1, how='all')\n",
    "\n",
    "    similar_user_books.drop(target_userid_read.columns,axis=1, inplace=True, errors='ignore')\n",
    "\n",
    "    #Prediction\n",
    "    book_score = {}\n",
    "    for i in similar_user_books.columns:\n",
    "        book_rating = similar_user_books[i]\n",
    "        total = 0\n",
    "        count = 0\n",
    "        for u in similar_user.index:\n",
    "            if pd.isna(book_rating[u]) == False:\n",
    "                score = similar_user[u] * book_rating[u]\n",
    "                total += score\n",
    "                count +=1\n",
    "        book_score[i] = total / count\n",
    "\n",
    "    book_score = pd.DataFrame(book_score.items(), columns=['book_id', 'book_score'])\n",
    "\n",
    "    ranked_book_score = pd.merge(book_score, books, on = 'book_id', how='inner')\n",
    "\n",
    "    ranked_book_score = ranked_book_score.sort_values(by='book_score', ascending=False)\n",
    "\n",
    "    avg_rating = user_book_matrix[user_book_matrix.index == userID].T.mean()[userID]\n",
    "\n",
    "    ranked_book_score['predicted_rating'] = ranked_book_score['book_score'] + avg_rating\n",
    "\n",
    "    ranked_book_score = ranked_book_score.drop(['book_score','predicted_rating'], axis=1)\n",
    "    ranked_book_score.rename(columns = {'book_id':'Book ID','title':'Title','authors':'Author','original_publication_year':'Publish Year'}, inplace=True)\n",
    "\n",
    "    return ranked_book_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b48aa7ec",
   "metadata": {
    "id": "b48aa7ec",
    "outputId": "030d639d-a397-4759-ab82-a53359631a7c"
   },
   "outputs": [],
   "source": [
    "userID = int(input(\"Enter user ID to whom you want to recommend : \"))\n",
    "\n",
    "while userID not in ratings['user_id'].unique():\n",
    "    userID = int(input(\"Enter valid userID: \"))\n",
    "\n",
    "recommendation = recommender_system(userID)\n",
    "print(f'\\nBook recommendations for user {userID}:')\n",
    "print(f'Total of Books: ', recommendation.shape[0])\n",
    "recommendation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f982dd5",
   "metadata": {
    "id": "1f982dd5",
    "outputId": "be83fb3c-49a7-4a78-ec6b-96415ac9d296"
   },
   "outputs": [],
   "source": [
    "print(f'\\nTop 10 book recommendations for user {userID}:')\n",
    "recommendation.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85e978ca",
   "metadata": {
    "id": "85e978ca"
   },
   "source": [
    "# Item-Based collaborative filtering ( Ooi Jin Kun )\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bf92928d",
   "metadata": {
    "id": "bf92928d"
   },
   "source": [
    "Defination :\n",
    "\n",
    "Item-based collaborative filtering is a technique used in recommender systems to provide personalized recommendations to users based on their interactions and preferences with items (such as products, movies, articles, etc.). It focuses on establishing relationships between items rather than users. The core idea behind item-based collaborative filtering is that if a user has shown a positive preference for one item, they are likely to have similar preferences for items that are closely related to it.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3dbe0794",
   "metadata": {
    "id": "3dbe0794"
   },
   "source": [
    "### Read user_id from user"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0d2fc9b",
   "metadata": {
    "id": "c0d2fc9b",
    "outputId": "51f8b9a5-c241-469f-879b-3896a06086d7"
   },
   "outputs": [],
   "source": [
    "target_user_id = int(input('Enter user_id : '))\n",
    "while target_user_id not in ratings['user_id'].unique():\n",
    "    target_user_id = int(input('Enter valid user_id : '))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "224184b1",
   "metadata": {
    "id": "224184b1"
   },
   "source": [
    "### Find user-books matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d6a60e4",
   "metadata": {
    "id": "9d6a60e4",
    "outputId": "8457b64a-76ce-49b7-cea9-87669f1daa1e"
   },
   "outputs": [],
   "source": [
    "ratings_books_matrix = ratings_books.pivot_table(index='user_id', columns=['book_id'],values = 'rating')\n",
    "ratings_books_matrix.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40c32e0a",
   "metadata": {
    "id": "40c32e0a"
   },
   "source": [
    "### Find book-user matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ec23111",
   "metadata": {
    "id": "5ec23111",
    "outputId": "d33b72aa-8aff-4399-e3f8-090f4b31104b"
   },
   "outputs": [],
   "source": [
    "books_ratings_matrix = ratings_books_matrix.transpose()\n",
    "books_ratings_matrix.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d8c3d73",
   "metadata": {
    "id": "0d8c3d73"
   },
   "source": [
    "### Find the information of user_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2ab6cd5",
   "metadata": {
    "id": "d2ab6cd5",
    "outputId": "da84f296-a454-4889-cc84-68427ad51291"
   },
   "outputs": [],
   "source": [
    "books_ratings_user = books_ratings_matrix[target_user_id]\n",
    "pd.DataFrame(books_ratings_user.sort_values(ascending= False)).rename(columns={target_user_id: f\"user_id {target_user_id}'s rating\"})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30738485",
   "metadata": {
    "id": "30738485"
   },
   "source": [
    "### Use user's highest rated books to recommend other related books"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e9f5538",
   "metadata": {
    "id": "0e9f5538",
    "outputId": "b4ffa6eb-6e68-435b-88c6-d537907652e4"
   },
   "outputs": [],
   "source": [
    "source_user_books_rating = ratings_books.loc[ratings_books['user_id'] == target_user_id].sort_values(by = 'rating',ascending = False)\n",
    "pd.DataFrame(source_user_books_rating.head(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23d04568",
   "metadata": {
    "id": "23d04568",
    "outputId": "8900c4cb-4d32-4a8e-a183-c368677426be"
   },
   "outputs": [],
   "source": [
    "top_rated_books_id = source_user_books_rating['book_id'].tolist()[0]\n",
    "top_rated_books_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b767b0b5",
   "metadata": {
    "id": "b767b0b5",
    "outputId": "1808d088-02c5-4003-8f6f-ff1fb755c98f"
   },
   "outputs": [],
   "source": [
    "ratings = ratings_books_matrix[top_rated_books_id]\n",
    "pd.DataFrame(ratings).rename(columns={top_rated_books_id: f\"{top_rated_books_id}'s rating\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f769b8b2",
   "metadata": {
    "id": "f769b8b2",
    "outputId": "3045738b-c859-4e8c-da9b-8ee8540ad32e"
   },
   "outputs": [],
   "source": [
    "similar_books = ratings_books_matrix.corrwith(ratings)\n",
    "similar_books = pd.DataFrame(similar_books, columns=['correlation'])\n",
    "similar_books"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7fd75a46",
   "metadata": {
    "id": "7fd75a46"
   },
   "source": [
    "### Identify the most correlated books"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97b7b680",
   "metadata": {
    "id": "97b7b680",
    "outputId": "6881d8ee-43ea-45c9-b16e-68d311331871"
   },
   "outputs": [],
   "source": [
    "sorted_similar_books = pd.DataFrame(similar_books, columns=['correlation']).sort_values(by= 'correlation', ascending= False)\n",
    "sorted_similar_books"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed6a4839",
   "metadata": {
    "id": "ed6a4839"
   },
   "source": [
    "### Eliminate the source books"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5103d48",
   "metadata": {
    "id": "b5103d48",
    "outputId": "5c48e368-d088-464d-b7b9-ef958d2e00e9"
   },
   "outputs": [],
   "source": [
    "sorted_similar_books = sorted_similar_books[1:]\n",
    "sorted_similar_books"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a56efcd0",
   "metadata": {
    "id": "a56efcd0"
   },
   "source": [
    "## Ensure the identified books is popular\n",
    "Higher number of votes means more popular"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0881d656",
   "metadata": {
    "id": "0881d656",
    "outputId": "c619a83d-694b-4eca-cb82-40c833c5bd38"
   },
   "outputs": [],
   "source": [
    "#Get number of rating for each books\n",
    "rating_votes = pd.DataFrame(ratings_books.groupby('book_id')['rating'].count())\n",
    "rating_votes=rating_votes.rename(columns={'rating': 'rating_count'})\n",
    "rating_votes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98787631",
   "metadata": {
    "id": "98787631",
    "outputId": "553d4027-76a9-4e3f-a6b1-fb8596ec17ca"
   },
   "outputs": [],
   "source": [
    "similar_books_ratings = sorted_similar_books.join(rating_votes['rating_count']).sort_values(by = 'correlation', ascending = False)\n",
    "similar_books_ratings"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c6b4381",
   "metadata": {
    "id": "3c6b4381"
   },
   "source": [
    "## Get the books that have higher votes and have higher correlation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7dd25e8",
   "metadata": {
    "id": "a7dd25e8",
    "outputId": "b180b3ac-8d84-408b-dfdb-6fecd13b26aa"
   },
   "outputs": [],
   "source": [
    "similar_popular_books = similar_books_ratings.loc[similar_books_ratings['rating_count']>=target_vote_number].dropna()\n",
    "similar_popular_books"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aac47cac",
   "metadata": {
    "id": "aac47cac",
    "outputId": "2bdec0d5-750d-4677-f3e9-f5ba7a18aebe"
   },
   "outputs": [],
   "source": [
    "#Find target_user's rated books\n",
    "target_user = ratings_books.loc[ratings_books['user_id'] == target_user_id].sort_values(by= 'rating', ascending= False)\n",
    "\n",
    "#trim the result exist in user's rating\n",
    "similar_popular_books = similar_popular_books[~similar_popular_books.index.isin(target_user['book_id'].tolist())]\n",
    "similar_popular_books = similar_popular_books.sort_values(by='correlation', ascending = False)\n",
    "similar_popular_books"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2770a9d",
   "metadata": {
    "id": "b2770a9d",
    "outputId": "049f4bae-bbef-4ad8-de8c-a8089d90b106"
   },
   "outputs": [],
   "source": [
    "#make it a list\n",
    "most_similar_popular_books = similar_popular_books[:numberOfResult]\n",
    "most_similar_popular_books_list = most_similar_popular_books.index.to_list()\n",
    "most_similar_popular_books_list"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7487419d",
   "metadata": {
    "id": "7487419d"
   },
   "source": [
    "### Show target book's information"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f911f83",
   "metadata": {
    "id": "4f911f83",
    "outputId": "72d43b9d-7a22-46b5-8a87-818f17223fba"
   },
   "outputs": [],
   "source": [
    "target_search_books = books.loc[books['book_id'] == top_rated_books_id]\n",
    "pd.DataFrame(target_search_books)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dbbfceb8",
   "metadata": {
    "id": "dbbfceb8"
   },
   "source": [
    "### Show books correlation table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b6bc12b4",
   "metadata": {
    "id": "b6bc12b4",
    "outputId": "4e8efb2b-de7a-4c98-cf6b-190ed824c9f4"
   },
   "outputs": [],
   "source": [
    "similar_popular_books"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9726b93c",
   "metadata": {
    "id": "9726b93c"
   },
   "source": [
    "## Item Based Recommended Result\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13a47a30",
   "metadata": {
    "id": "13a47a30",
    "outputId": "00c6133d-f67c-4a3c-e238-9b4666c72c93"
   },
   "outputs": [],
   "source": [
    "# Item-based recommended result\n",
    "most_similar_popular_books_df = books.loc[books['book_id'].isin(most_similar_popular_books_list)]\n",
    "most_similar_popular_books_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "922f2b14",
   "metadata": {
    "id": "922f2b14"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
