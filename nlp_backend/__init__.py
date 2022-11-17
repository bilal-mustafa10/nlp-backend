from ast import literal_eval

import pandas as pd
from tqdm.notebook import tqdm

tqdm.pandas()
from transformers import pipeline

data_path = "nlp_backend/assets/data.csv"
df = pd.read_csv(data_path, sep=',', converters={'doc_entities': literal_eval, 'doc_keyphrases': literal_eval})
df.dropna()
df['sentence'] = df['sentence'].fillna('')
interesting = df.drop(['doc_date', 'doc_title', 'doc_entities', 'doc_keyphrases', 'doc_publish_location'], axis=1)
interesting.dropna()
interesting['sentence'] = interesting['sentence'].fillna('')
model_name = "deepset/roberta-base-squad2"
fb_ai = pipeline('question-answering', model=model_name, tokenizer=model_name)