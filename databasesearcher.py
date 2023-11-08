import pandas as pd

import numpy as np

import argparse

from ast import literal_eval

import openai


# Define constants for the application

EMBEDDING_MODEL = "text-embedding-ada-002"

MAX_TOKENS = 8000


# Function to load dataset and check for embeddings

def load_dataset(input_path):

    """Loads the dataset and checks if embeddings exist.

    

    Args:

        input_path (str): Path to the dataset file.

        

    Returns:

        DataFrame: The loaded dataset with or without existing embeddings.

    """

    # Load dataset and attempt to parse embeddings as lists if they exist

    df = pd.read_csv(input_path, converters={'embedding': literal_eval}, na_values=['None'])

    

    # Determine if embeddings column has valid entries

    embeddings_exist = not df['embedding'].isnull().all()

    return df, embeddings_exist


# Function to generate embeddings for dataset if needed

def generate_embeddings(df):

    """Generates embeddings for the dataset if they do not exist.

    

    Args:

        df (DataFrame): The dataset DataFrame.

        

    Returns:

        DataFrame: The dataset with generated embeddings.

    """

    # If embeddings do not exist, create them using the get_embedding function

    if 'embedding' not in df.columns or df['embedding'].isnull().all():

        df['embedding'] = df['content'].apply(lambda x: get_embedding(x) if isinstance(x, str) else np.nan)

    return df


# Function to search through the dataset using embeddings

def search_dataset(df, query, top_n=3):

    """Searches through the dataset using embedding similarity.

    

    Args:

        df (DataFrame): The dataset with embeddings.

        query (str): The search query to find similar content.

        top_n (int): The number of top similar results to return.

        

    Returns:

        DataFrame: The top similar results to the query.

    """

    # Convert query to embedding vector

    query_embedding = get_embedding(query)

    

    # Compute cosine similarity and sort results

    df['similarity'] = df['embedding'].apply(lambda emb: cosine_similarity(emb, query_embedding))

    top_results = df.nlargest(top_n, 'similarity')

    

    return top_results


# Argument parser setup

parser = argparse.ArgumentParser(description="Search PDF Embeddings")

parser.add_argument('file_path', help='Path to the dataset file.')

parser.add_argument('query', help='Search query.')


# Parse command-line arguments

args = parser.parse_args()


# Main program execution

if __name__ == "__main__":

    # Load the dataset and check for existing embeddings

    dataset, embeddings_exist = load_dataset(args.file_path)

    

    # Generate embeddings if they do not exist

    if not embeddings_exist:

        print("Embeddings not found in dataset, generating...")

        dataset = generate_embeddings(dataset)

    

    # Perform search on the dataset with the query

    results = search_dataset(dataset, args.query)

    

    # Display top search results

    for index, row in results.iterrows():

        print(f"ID: {row['id']}, Content: {row['content']}, Similarity: {row['similarity']}")

        

    # Future optimization: Implement batch processing for generating embeddings to minimize API calls.

    # Future optimization: Use vectorized operations for similarity calculation instead of apply.
