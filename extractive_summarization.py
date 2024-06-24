# Perform necessary imports
import pandas as pd

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


def extractive_summarization(text, tokenizer, input_limit=1024):
    parser = PlaintextParser.from_string(text, Tokenizer("dutch"))
    summarizer = TextRankSummarizer()

    for k in reversed(range(30)):
        tokenized_text = tokenizer(text, return_tensors="pt", truncation=False)
        if tokenized_text["input_ids"].shape[1] < input_limit:
            summary = summarizer(parser.document, 100)
            summary_list = [str(sentence) for sentence in summary]
            summary_text = " ".join(summary_list)
            break
        else:
            summary = summarizer(parser.document, k)
            summary_list = [str(sentence) for sentence in summary]
            summary_text = " ".join(summary_list)

            tokenizer_summary = tokenizer(summary_text, return_tensors="pt")
            
            if tokenizer_summary["input_ids"].shape[1] < input_limit:
                break
    
    return summary_text