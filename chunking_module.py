# Perform necessary imports
import numpy as np
import pandas as pd
import spacy
from rouge_score import rouge_scorer
from collections import defaultdict

# Load the Dutch language model
nlp = spacy.load("nl_core_news_md")


# Define function that creates chunks from a document
#####################################################################################################
def split_text_into_chunks(text, max_token_length, split_method="newline"):
    """ Note: This function takes in a piece of text, and creates individual chunks from it.
        Note that not only a single text, but also individual paragraphs, and even individual sentences might exceed the input_limit
        Note that for this function to work properly, no single sentence within a case-text should be of a length greater than max_token_length (for our purpose, that will most likely not be the case)"""
    chunks = []
    current_chunk = ""

    if split_method == "newline":
        # Split text into paragraphs based on "\n" tokens. This is done at first.
        paragraphs = text.split("\n")

    if split_method == "sent_tokenize":
        # Split text into sentences. Only done if a single paraphraph is longer than the max_token_length
        doc = nlp(text)
        paragraphs = [sent.text for sent in doc.sents]
    
    # Loop through the paragraphs/chunks and perform the actual chunking
    for paragraph in paragraphs:
        # Check if the current paragraph is longer than the maximum token length
        if len(paragraph) > max_token_length:
            if split_method == "newline":
                # When a single paraphraph is longer than the max_token_length, split the paragraph into sentences
                chunks_par = split_text_into_chunks(paragraph, max_token_length, split_method="sent_tokenize")
                for chunk in chunks_par:
                    chunks.append(chunk)

        else:
            if len(current_chunk) + len(paragraph) <= max_token_length:
            # Check if adding the current paragraph exceeds the input limit
                current_chunk += paragraph + " "
            else:
                chunks.append(current_chunk.strip())
                # Start a new chunk with the current paragraph
                current_chunk = paragraph + " "

    # Add the last chunk to the list of chunks
    chunks.append(current_chunk.strip())

    return chunks


# Define function that generates a ROUGE matrix
#####################################################################################################
def generate_rouge_matrix(text_chunks, summary_sentences, rouge_metric):
    # Initialize RougeScorer
    scorer = rouge_scorer.RougeScorer(rouge_types = [rouge_metric], use_stemmer=True)
    
    # Initialize matrix to store ROUGE scores
    rouge_matrix = np.zeros((len(summary_sentences), len(text_chunks)))
    
    # Calculate ROUGE scores for each pair of text chunk and summary sentence
    for i, sentence in enumerate(summary_sentences):
        for j, chunk in enumerate(text_chunks):
            scores = scorer.score(chunk, sentence)          # Note, in scorer.score(x,y), x corresponds to the reference summary, and y corresponds to the generated summary
            rouge_score = scores[rouge_metric].precision    # By using precision, we calculate the rouges-score with regard to the sentence itself (i.e., rouge = n-gram-overlap/n-gram-total-sentence)
            rouge_matrix[i][j] = rouge_score
    
    return rouge_matrix


# Define function that finds the most similar chunk for each summary sentence
#####################################################################################################
def find_most_similar_chunk(rouge_matrix):
    # Find the index of the chunk with the highest ROUGE score for each summary sentence
    most_similar_chunk_indices = np.argmax(rouge_matrix, axis=1)
    return most_similar_chunk_indices


# Define function that creates a dataframe from the chunks and summary sentences
#####################################################################################################
def chunk_summary_assignment_dict(summary_sentences, most_similar_chunk_indices):
    # Create a dictionary to store sentences associated with each chunk
    chunk_sentences_dict = {}

    # Populate the dictionary with sentences associated with each chunk
    for j, summary_sentence in enumerate(summary_sentences):
        most_similar_chunk_index = most_similar_chunk_indices[j]
        if most_similar_chunk_index not in chunk_sentences_dict:
            chunk_sentences_dict[most_similar_chunk_index] = []
        chunk_sentences_dict[most_similar_chunk_index].append(summary_sentence)

    # Create a list to store chunk-summary pairs
    chunk_summary_pairs = {}

    # Concatenate sentences associated with each chunk
    for chunk_index, sentences in chunk_sentences_dict.items():
        summary_sentence = " ".join(sentences)  # Concatenate sentences associated with the chunk
        chunk_summary_pairs[chunk_index] = summary_sentence

    return chunk_summary_pairs


#Let's combine all the functions into a single function
#####################################################################################################
def create_chunk_summary_df(ecli, text, summary, max_token_length=4096, rouge_metric='rouge2', split_method='newline'):
    """ Function that takes in a piece of text and a summary, and returns a dataframe with the most similar chunk for each summary sentence.
        Default values: 
        - max_token_length=4096
        - rouge_metric='rouge2'
        - split_method='newline'
    """
    # Split the text into chunks
    text_chunks = split_text_into_chunks(text, max_token_length, split_method=split_method)
    num_original_chunks = len(text_chunks)

    # Add the text chunks to a new dataframe
    df_chunks = pd.DataFrame(text_chunks, columns=['text_chunk'])
    df_chunks['ecli'] = ecli
    df_chunks['chunk_id'] = df_chunks.index
    
    # Split the summary into sentences
    doc = nlp(summary)
    summary_sentences = [sent.text for sent in doc.sents]
    
    # Generate a ROUGE matrix
    rouge_matrix = generate_rouge_matrix(text_chunks, summary_sentences, rouge_metric=rouge_metric)
    
    # Find the most similar chunk for each summary sentence
    most_similar_chunk_indices = find_most_similar_chunk(rouge_matrix)
    
    # Create a dataframe from the chunks and summary sentences
    chunk_id_sum_dict = chunk_summary_assignment_dict(summary_sentences, most_similar_chunk_indices)

    # Add the summary sentences to the dataframe
    df_chunks['summary_sentence'] = df_chunks['chunk_id'].map(chunk_id_sum_dict)


    # Apply some post-processing
    # Drop duplicate rows, just in case
    df_chunks = df_chunks.drop_duplicates(subset=["ecli", "text_chunk", "summary_sentence"])

    # Compute the distribution ratio
    # First, perform some cleaning
    df_chunks["summary_sentence"] = df_chunks["summary_sentence"].str.replace("\n", "").replace('', np.nan)
    df_chunks.loc[df_chunks["summary_sentence"].str.len() < 20, "summary_sentence"] = np.nan
    df_chunks["summary_sentence"] = df_chunks["summary_sentence"].fillna("<NoSum>")

    df_chunks["num_original_chunks"] = num_original_chunks
    df_chunks["num_chunks_final"] = (df_chunks['summary_sentence'] != "<NoSum>").sum()

    df_chunks["distribution_ratio"] = df_chunks["num_chunks_final"]/df_chunks["num_original_chunks"]

    # Reorder the columns
    df_chunks = df_chunks[['ecli', 'chunk_id', 'text_chunk', 'summary_sentence', 'num_original_chunks', 'num_chunks_final', 'distribution_ratio']]


    return df_chunks

