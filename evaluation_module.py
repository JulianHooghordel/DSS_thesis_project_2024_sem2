import pandas as pd
import numpy as np

from rouge_score import rouge_scorer


def generate_summary(text, model, tokenizer, max_gen_length, max_input_length):
    """Function to generate a summary given an input text"""
    # Tokenize the text_chunk
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_gen_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def evaluate_summary(reference_summary, generated_summary):
    """Function to evaluate the generated summary using the ROUGE metric"""
    # Calculate the ROUGE-score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)

    # Extrac the f1-scores
    f1_scores = [round(score.fmeasure, 4) for score in scores.values()]

    return f1_scores


def compute_summary_dataframe(dataframe, model, tokenizer, max_input_length, type):
    """Note that for the extraction and chunking methods, we use differen colum names!"""

    if type == "text_chunk":
        max_gen_length = 600
        # Generate the summaries
        dataframe["generated_summary_sentences"] = dataframe["text_chunk"].apply(lambda x: generate_summary(x, model, tokenizer, max_gen_length, max_input_length))
        dataframe["generated_summary_sentences"] = dataframe["generated_summary_sentences"].apply(lambda x: x.replace("<NoSum>", ""))
        dataframe = dataframe.groupby("ecli").agg({"text_chunk": " ".join, "summary_sentence": " ".join, "generated_summary_sentences": " ".join}).reset_index()
        # Rename columns
        dataframe.rename(columns={"text_chunk": "full_text", "summary_sentence": "reference_summary", "generated_summary_sentences": "generated_summary"}, inplace=True)
        

    elif type == "extractive_summary":
        max_gen_length = 2048
        # Generate the summaries
        dataframe["generated_summary"] = dataframe["extractive_summary"].apply(lambda x: generate_summary(x, model, tokenizer, max_gen_length, max_input_length))
        # Rename columns
        dataframe.rename(columns={"inhoudsindicatie": "reference_summary"}, inplace=True)

    else:
        raise ValueError("The type parameter must be either 'text_chunk' or 'extractive_summary'")
    
    return dataframe