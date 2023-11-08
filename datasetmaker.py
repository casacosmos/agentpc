import openai

import pandas as pd

import json

import os

import argparse

from transformers import GPT2Tokenizer

from PyPDF2 import PdfReader

import nltk


nltk.download('punkt') # Ensure the punkt tokenizer models are downloaded


# Define constants for the application

EMBEDDING_MODEL = "text-embedding-ada-002"


# Function to read and extract text from a PDF file.

def read_pdf(filepath):

    """Reads a PDF and extracts text content.


    Args:

        filepath (str): File path of the PDF document to be read.


    Returns:

        str: Extracted text content from the PDF.


    Raises:

        Exception: If the file is not accessible or readable.

    """

    try:

        reader = PdfReader(filepath)

        pdf_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

        return pdf_text

    except (FileNotFoundError, OSError) as e:

        raise Exception(f"Error reading the file: {e}")


# Function to truncate text length to a specified maximum length.

def reduce_long(text, max_len=590):

    """Reduces text length to a specified maximum length.


    Args:

        text (str): The text to be reduced.

        max_len (int): Maximum token length for reducing text.


    Returns:

        str: Reduced text.

    """

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokens = tokenizer.encode(text)

    if len(tokens) <= max_len:

        return text
    

    sentences = nltk.sent_tokenize(text)

    while len(tokens) > max_len and sentences:

        sentences.pop()

        reduced_text = ' '.join(sentences)

        tokens = tokenizer.encode(reduced_text)

    return reduced_text


# Function to tokenize text into sentences.

def sentence_tokens(text):

    """Tokenizes text into sentences.


    Args:

        text (str): Text to be tokenized into sentences.


    Returns:

        list of str: A list of sentences from the text.

    """

    return nltk.sent_tokenize(text)


# Function to generate an embedding vector for the given text content.

def get_embedding(text):

    """Generates an embedding vector for the given text content.


    Args:

        text (str): Text to generate embeddings for.


    Returns:

        list: Embedding vector for text.


    Raises:

        Exception: If there's an error in generating embeddings.

    """

    try:

        response = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)

        embedding = response['data'][0]['embedding']

        return embedding

    except openai.error.OpenAIError as e:

        raise Exception(f"Error generating embeddings: {e}")


# Function to prepare a dataset comprising sentence embeddings.

def prepare_dataset(filepath):

    """Prepares a dataset comprising sentence embeddings.


    Args:

        filepath (str): File path of the PDF document to be processed.


    Returns:

        list of dict: Data points containing ids, content, embeddings, and metadata.

    """

    text = read_pdf(filepath)

    sentences = sentence_tokens(text)

    data = [{

        "id": idx,

        "content": sentence,

        "embedding": get_embedding(sentence),

        "metadata": {}  # Placeholder for additional metadata

    } for idx, sentence in enumerate(sentences, 1)]

    return data


# Function to save the data to a JSON file.

def save_to_json(data, output_filename):

    """Saves the data to a JSON file.


    Args:

        data (list): Data to be saved.

        output_filename (str): Output filename for the JSON.

    """

    with open(output_filename, 'w') as f:

        json.dump(data, f)

    print(f"Dataset saved to {output_filename}")


# Function to save the data to a CSV file.

def save_to_csv(data, output_filename):

    """Saves the data to a CSV file.


    Args:

        data (list): Data to be saved.

        output_filename (str): Output filename for the CSV.

    """

    df = pd.DataFrame(data)

    df['embedding'] = df['embedding'].apply(json.dumps)  # Convert embedding list to JSON string

    df['metadata'] = df['metadata'].apply(json.dumps)    # Convert metadata dict to JSON string

    df.to_csv(output_filename, index=False)

    print(f"Dataset saved to {output_filename}")


# Argument parser setup

parser = argparse.ArgumentParser(description="PDF Embedding Extractor")

parser.add_argument("filepath", help="Filepath of the PDF document to be processed.")

args = parser.parse_args()


# Main program execution

if __name__ == "__main__":

    dataset = prepare_dataset(args.filepath)

    filename_no_ext = os.path.splitext(os.path.basename(args.filepath))[0]

    json_filename = f"{filename_no_ext}.json"

    csv_filename = f"{filename_no_ext}.csv"

    save_to_json(dataset, json_filename)

    save_to_csv(dataset, csv_filename)


# Here the creation of the embedding and the data files is decoupled.

# The get_embedding function can be optimized by batch processing in future.

# The tokenizer does not handle multiple languages; add support if needed.
