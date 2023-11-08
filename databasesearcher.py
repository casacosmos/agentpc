import pandas as pd

import numpy as np

import argparse

from ast import literal_eval

import openai


# Initialization of OpenAI API key should ideally be handled more securely or outside of the code.

openai.api_key = 'your-api-key'


# Function to generate embeddings for a given text.

def get_embedding(text):

    if not isinstance(text, str):

        raise ValueError("Input must be a string")

    

    preprocessed_text = text  # Assumes API handles preprocessing, but could be more explicit.


    try:

        response = openai.Embedding.create(

            input=preprocessed_text,

            model="text-embedding-ada-002"

        )

        embedding = response['data'][0]['embedding']

    except Exception as e:

        raise Exception("Error generating embedding: " + str(e))


    embedding_list = np.array(embedding).tolist()


    return embedding_list


# Function should include a more robust method for parsing embeddings, to avoid incorrect operations on malformed data.

def load_dataset(input_path):

    df = pd.read_csv(input_path, converters={'embedding': literal_eval}, na_values=['None'])


    embeddings_exist = not df['embedding'].isnull().all()


    return df, embeddings_exist


# This function can be enhanced using vectorized operations or parallel processing.

def generate_embeddings(df):

    if 'embedding' not in df.columns or df['embedding'].isnull().all():

        df['embedding'] = df['content'].apply(lambda x: get_embedding(x) if isinstance(x, str) else np.nan)


    return df


# Optimization can be done using vectorized operations or scikit-learn utilities for similarity calculation.

def search_dataset(df, query, top_n=3):

    query_embedding = get_embedding(query)


    # Instantiate a cosine_similarity function, vectorized operations can lead to significant speed up.

    df['similarity'] = df['embedding'].apply(lambda emb: cosine_similarity(emb, query_embedding))


    top_results = df.nlargest(top_n, 'similarity')


    return top_results


parser = argparse.ArgumentParser(description="Search PDF Embeddings")

parser.add_argument('file_path', help='Path to the dataset file.')

parser.add_argument('query', help='Search query.')


args = parser.parse_args()


if __name__ == "__main__":

    dataset, embeddings_exist = load_dataset(args.file_path)


    if not embeddings_exist:

        print("Embeddings not found in dataset, generating...")

        dataset = generate_embeddings(dataset)


    results = search_dataset(dataset, args.query)


    for index, row in results.iterrows():

        # Provide formatted output for better readability.

        print(f"ID: {row['id']}, Content: {row['content']}, Similarity: {row['similarity']}")


    # Future optimization could include batch embedding generation to minimize API usage.

    # Future optimization: pre-calculate and store similarity scores for faster search operations.
