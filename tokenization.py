import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from datasets import load_from_disk

#model constants
MODEL_NAME = 'roberta-base'
MAX_LEN = 256 #revisit later if accuracy is low!

#loading cleaned data
train_df = pd.read_csv('data/train_clean.csv')
valid_df = pd.read_csv('data/valid_clean.csv')
test_df = pd.read_csv('data/test_clean.csv')

#Convert to Huggingface Datasets
ds = DatasetDict({
    'train': Dataset.from_pandas(train_df, preserve_index=False),
    'validation': Dataset.from_pandas(valid_df, preserve_index=False),
    'test': Dataset.from_pandas(test_df, preserve_index=False)
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch['input_text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN
    )

ds = ds.map(tokenize, batched=True)

keep = ["id", "input_ids", "attention_mask", "label"]
cols_to_remove = [c for c in ds["train"].column_names if c not in keep]
ds = ds.remove_columns(cols_to_remove)

#saves it to disk and creates a file that can be loaded later for model training
ds.save_to_disk('data/tokenized_datasets')

print(ds)