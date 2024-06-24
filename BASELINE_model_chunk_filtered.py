# Make sure to disable parallelism for the tokenizers library, as it may cause issues with the training process
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# BASELINE model, trained on complete dataset, chunking approach
#Perform the necessary imports
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
import torch
import evaluate
import numpy as np

if torch.cuda.is_available():
    print("CUDA is available. Number of GPUs:", torch.cuda.device_count())
    device = torch.device('cuda')



####################################################################################################################################################################################################################
# Define parameters
####################################################################################################################################################################################################################
full_df_path = "data/dataset_1990_sum_big_complete_chunk_3072.csv"
complete_eclis_path = "data/ecli_series/ecli_dataset_1990_sum_filtered.csv"

text_column = "text_chunk"
summary_column = "summary_sentence"

output_dir = "models/BASELINE_UL2_chunk_filtered"



####################################################################################################################################################################################################################
# First, load the initial data. Perform filtering based on the selected ECLI's. Load everything into a DatasetDict object.
####################################################################################################################################################################################################################
df = pd.read_csv(full_df_path)
complete_eclis = pd.read_csv(complete_eclis_path).iloc[:, 0]

test_eclis = pd.read_csv("data/ecli_series/ECLI_CHUNK_FILTERED_TEST.csv").iloc[:, 0]
val_eclis = pd.read_csv("data/ecli_series/ECLI_CHUNK_FILTERED_VAL.csv").iloc[:, 0]
train_eclis = complete_eclis[~complete_eclis.isin(test_eclis) & ~complete_eclis.isin(val_eclis)]

print(">>> Shape of the original dataset: ", df.shape)
print(">>> Shape of the filtered ECLI's: ", complete_eclis.shape)

df = df[df["ecli"].isin(complete_eclis)]
print(">>> Shape of the filtered dataset: ", df.shape)

assert complete_eclis.shape[0] == df["ecli"].unique().shape[0]

# Drop the rows with None values in the text_chunk column
df = df.dropna(subset=['text_chunk'])

# Split the data into a training, validation and test set
train_df = df[df["ecli"].isin(train_eclis)]
val_df = df[df["ecli"].isin(val_eclis)]
test_df = df[df["ecli"].isin(test_eclis)]

print(">>> Shape of train dataset: ", train_df.shape)
print(">>> Shape of validation dataset: ", val_df.shape)
print(">>> Shape of test dataset: ", test_df.shape)

assert train_df.shape[0] + val_df.shape[0] + test_df.shape[0] == df.shape[0]


#Lets convert the data into a Dataset object, and put them into a single DatasetDict object
train_dataset = Dataset.from_pandas(train_df[["ecli", text_column, summary_column]])
val_dataset = Dataset.from_pandas(val_df[["ecli", text_column, summary_column]])
test_dataset = Dataset.from_pandas(test_df[["ecli", text_column, summary_column]])

rechtspraak_dataset = DatasetDict({
    "train": train_dataset, 
    "validation": val_dataset, 
    "test": test_dataset
})


####################################################################################################################################################################################################################
# Load the tokenizer and model
####################################################################################################################################################################################################################
model_name = "yhavinga/ul2-base-dutch"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name) 

# Move the model to the GPU
model.to(device)


####################################################################################################################################################################################################################
# Let's update the model configuration
####################################################################################################################################################################################################################
# By default, the model is able to generate output texts with a max_length of 200 tokens (model.generation_config). However, for our case, we want it to be able to generate longer sequences
print(model.generation_config)
new_max_length = 600
model.generation_config.max_length = new_max_length


####################################################################################################################################################################################################################
# Let's tokenize the dataset
####################################################################################################################################################################################################################
prefix = "Vat samen: "    # This is the prefix that the model will use to understand that it should generate a summary of the input text. This is a standard prefix for T5 models, and it is important to include it in the input text.
def process_data_to_model_inputs(batch):
    input_texts = [prefix + text for text in batch[text_column]]                                            # We add the prefix to the input text
    model_inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=768)        	# To tokenize the input text
    labels = tokenizer(batch[summary_column], padding="max_length", truncation=True, max_length=768)        # To tokenize the input summary. We call these "labels" because these essentially represent the thing the model is supposed to predict
    model_inputs["labels"] = labels["input_ids"]    # We add a 'labels' key to the model_inputs, which correspond to the reference summary. We do not include the attention mask of this summary, because this is simply not necessary.
    
    return model_inputs

# Let's process the data
rechtspraak_dataset_tokenized = rechtspraak_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=8,
)

# We are going to create a DataCollator object
# This will be used to create batches for training
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)



####################################################################################################################################################################################################################
# Let's define the evaluation metric
####################################################################################################################################################################################################################
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred                                                                 # eval_pred is a tuple containing the predictions (the predicted summaries) and the labels (the reference summaries)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)                   # We decode the predictions (i.e., the summaries)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)                               # If labels do not equal -100, return them, else, return the padding token id. Note that the value of -100 is a placeholder
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)  # This generates the rouge1, rouge2, rougeL and rougeLSum scores
                                                                                                    # Note that result is a dictionary

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]    # Let's compute the lengths of the predicted summaries
    result["gen_len"] = np.mean(prediction_lens)            # Compute the average length of the generated summaries

    return {k: round(v, 4) for k, v in result.items()}      # The scores are rounded to 4 decimal points. Remember that result is a dictionary with keys corresponding to the score types



####################################################################################################################################################################################################################
# Now, let's train the model
####################################################################################################################################################################################################################
my_output_dir = output_dir

# Let's define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=my_output_dir,                           # This is the only required hyperparameter. It specifies the output directory of the model
    overwrite_output_dir=True,
    eval_strategy="no",                                 # No evaluation during training
    learning_rate=3e-5,                                 # Learning rate for the AdamW optimizer
    per_device_train_batch_size=8,                      # The batch size for training examples per decive (GPU)
    per_device_eval_batch_size=8,                       # The batch size for evaluation examples per decive (GPU)
    gradient_accumulation_steps=4,                      # Number of updates steps to accumulate before performing a backward/update pass
    weight_decay=0.01,                                  # Coefficient for regularization
    save_total_limit=3,                                 # The maximum number of checkpoints that can be saved
    num_train_epochs=4,                                 # Number of training epochs
    predict_with_generate=True,                         # Specifies whether to generate predictions during evaluation.
    generation_config = model.generation_config,        # To make sure that the model is able to generate up to 600 tokens
    generation_max_length = model.generation_config.max_length,			            # For safety, if generation_config does not work appropriately
    fp16=False,                                          # Whether to use 16-bit (also known as half- or mix-precision) floating-point numbers for training to speed up computations.
    push_to_hub=False,                                  # Wheter to push the trained model to the HuggingFace model hub after training
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=rechtspraak_dataset_tokenized["train"],
    eval_dataset=rechtspraak_dataset_tokenized["validation"],       # We will perform training and validation multiple times, given different hyperparameters
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,                                # Let's use our own custom ROUGE-metric scorer function
)

print(">>> Let's start model training!")
trainer.train()    			# This will start the training process. The trainer will iterate over the training data for the specific number of epochs, updating the model parameters based on the optimization algorithm



####################################################################################################################################################################################################################
# Let's save the model
####################################################################################################################################################################################################################
print(">>> Let's save the model")
model.save_pretrained(my_output_dir)		            # Since we do not push our model to the hub, we will have to save it locally in order to actually re-use it later on
model.generation_config.save_pretrained(my_output_dir)	# Also save the generation configuration
model.config.save_pretrained(my_output_dir)             # Also save the general model configuration
tokenizer.save_pretrained(my_output_dir)                # Also save the tokenizer

print(">>> End of script")
