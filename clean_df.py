#Running this file cleans the Kaggle dataset and splits it into train, valid, test sets

import pandas as pd 
from sklearn.model_selection import train_test_split
import re
import hashlib
#LIAR datasets used for independent robustness testing
liar_test = pd.read_csv('LIAR/test.tsv', sep='\t', header=None) #liar test has 1267 samples --> 9.905% of total data
liar_train = pd.read_csv('LIAR/train.tsv', sep='\t', header=None) #liar train has 10240 samples --=> 80.05% of total data
liar_valid = pd.read_csv('LIAR/valid.tsv', sep='\t', header=None) #liar valid has 1284 samples --> 10.038% of total data

#Currently all unused in model --> potentially to be added as additional feature later for social contet
gossipcop_fake = pd.read_csv('FAKENewsNet/gossipcop_fake.csv') #gossipcop_fake has 5323 samples --> 22.94% of total data
gossipcop_real = pd.read_csv('FAKENewsNet/gossipcop_real.csv') #gossipcop_real has 16817 samples --> 72.49% of total data
politifact_fake = pd.read_csv('FAKENewsNet/politifact_fake.csv') #politifact_fake has 432 samples --> 1.86% of total data
politifact_real = pd.read_csv('FAKENewsNet/politifact_real.csv') #politifact_real has 624 samples --> 2.69% of total data

#Kaggle Dataset used for main model
kaggle_fake = pd.read_csv('Kaggle Dataset/Fake.csv') #23481 samples --> 52.29% of total data
kaggle_real = pd.read_csv('Kaggle Dataset/True.csv') #21417 samples --> 47.70% of total data

#merging Kaggle Dataset into one
kaggle_fake['label'] = 1
kaggle_real['label'] = 0
main_df = pd.concat([kaggle_fake, kaggle_real], ignore_index=True)

# create a stable id column BEFORE splitting
main_df = main_df.reset_index(drop=True)
main_df["id"] = main_df.index

#splitting main dataset into train, test, valid
train_df, temp_df = train_test_split(main_df, 
                                     test_size=0.3, 
                                     random_state=42, 
                                     stratify=main_df['label'])
valid_df, test_df = train_test_split(temp_df, 
                                      test_size=0.50, 
                                      random_state=42,
                                      stratify=temp_df['label'])

#text cleaning main dataset
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<.*?>", " ", text)  # Remove HTML tags
    text = re.sub(r"http\S+", " " , text)  # Remove URLs
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    return text.strip()
for df in [train_df, valid_df, test_df]:
    df['title'] = df['title'].apply(clean_text)
    df['text'] = df['text'].apply(clean_text)
for df in [train_df, valid_df, test_df]:
    df["input_text"] = "TITLE: " + df["title"] + " BODY: " + df["text"]
    
#creating hash ids based on input_text
def make_hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

for df in [train_df, valid_df, test_df]:
    df["id"] = df["input_text"].apply(make_hash_id)

#saving all cleaned datasets to a file so they can be loaded later for tokenization
train_df.to_csv('data/train_clean.csv', index=False)
valid_df.to_csv('data/valid_clean.csv', index=False)
test_df.to_csv('data/test_clean.csv', index=False)