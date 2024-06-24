from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

import spacy
import pandas as pd

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset



############################################################################################################
# Load the model and tokenizer
############################################################################################################
model = AutoModelForSeq2SeqLM.from_pretrained("yhavinga/ul2-base-dutch")
tokenizer = AutoTokenizer.from_pretrained("yhavinga/ul2-base-dutch")
print("Tokenizer is fast: ", tokenizer.is_fast)

# read in a text file that contains tokens on every new line
with open("new_tokens_3.txt", "r") as file:
    new_tokens = file.readlines()

new_tokens = [token.strip() for token in new_tokens]
print(new_tokens[:10])
print(">>> Number of new tokens", len(new_tokens))

print(">>> Old length of the tokenizer", len(tokenizer))

tokenizer.add_tokens(new_tokens)
print(">>> Lenght of the tokenizer: ", len(tokenizer))

model.resize_token_embeddings(len(tokenizer))
print(">>> Lenght of the tokenizer:", len(tokenizer))
print(">>> Size of the embeddings:", model.get_input_embeddings().weight.shape[0])



############################################################################################################
# Define relevant functions
############################################################################################################
def split_text_into_chunks(text, max_token_length = 4000):
    # Load the Dutch language model
    nlp = spacy.load("nl_core_news_md")
    chunks = []
    current_chunk = ""
    
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    for sent in sentences:
        if len(current_chunk) + len(sent) <= max_token_length:
            current_chunk += sent + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent + " "

    return chunks


def mask_important_sentences_with_sentinels(text, ratio=0.25):
    parser = PlaintextParser.from_string(text, Tokenizer("dutch"))
    summarizer = TextRankSummarizer()
    num_sentences_to_mask = int(len(parser.document.sentences) * ratio)
    #print(f"Initial number of sentences: {len(parser.document.sentences)}")
    #print(f"Number of sentences to mask: {num_sentences_to_mask}")
    summary = summarizer(parser.document, num_sentences_to_mask)
    
    # Mask important sentences with sentinel tokens
    important_sentences = [str(sentence) for sentence in summary]
    for i, sentence in enumerate(important_sentences):
        sentinel_token = f"<extra_id_{i}>"
        text = text.replace(sentence, sentinel_token)

    prefix = "[NLG]"
    text = f"{prefix} {text}"
    
    return text, important_sentences


def custom_collate_fn(batch, tokenizer):
    text_string = " ".join(batch["full_text"])
    masked_texts = []
    labels = []


    # Chunk the text into smaller parts
    chunks = split_text_into_chunks(text_string)
    #print(">>> Number of chunks:", len(chunks))

    for chunk in chunks:
        masked_text, masked_sentences = mask_important_sentences_with_sentinels(chunk)
        masked_texts.append(masked_text)
        label = " ".join([f"<extra_id_{i}> {sentence}" for i, sentence in enumerate(masked_sentences)])
        labels.append(label)

    dataset_dict = {
        'input_text': masked_texts,
        'labels': labels,
    } 

    return dataset_dict



############################################################################################################
# Load the dataset
############################################################################################################
# First, load the full texts
path = "data/dataset_1990_no_sum.csv"
df = pd.read_csv(path)
df = df.sample(100)

# Tokenize the texts
dataset = Dataset.from_pandas(df)
print(">>> Dataset created")
print(dataset)

# Apply the custom processing function to each example in the dataset
processed_dataset = dataset.map(lambda batch: custom_collate_fn(batch, tokenizer), batched=True, batch_size=50, remove_columns=dataset.column_names)
print(">>> Processed dataset created")
print(processed_dataset)

processed_dataset.save_to_disk("data/GSG_processed_data_TEST2")


# Split the dataset into training and testing sets
train_test_split = processed_dataset.train_test_split(test_size=0.05)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']



############################################################################################################
# Define a data collator for dynamic padding
############################################################################################################
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



############################################################################################################
# Define the training arguments
############################################################################################################
print(">>> Start training")
output_dir = "models/UL2-base-dutch-rechtspraak"

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# Train the model
#trainer.train()



############################################################################################################
# Save the model
############################################################################################################
#print(">>> Training finished. Saving the model.")
#model.save_pretrained(output_dir)

print(">>> End of script.")
