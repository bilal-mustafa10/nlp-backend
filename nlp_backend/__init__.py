import pandas as pd
from ast import literal_eval
from tqdm.notebook import tqdm
tqdm.pandas()
from datetime import datetime
from transformers import pipeline


data_path = "../assets/cleanedDF.csv"
df = pd.read_csv(data_path, sep='\t', converters={'doc_entities': literal_eval, 'doc_keyphrases': literal_eval})
interesting = df.drop(['id','doc_date', 'doc_title', 'doc_url', 'doc_entities', 'doc_keyphrases', 'doc_publish_location'], axis=1)
model_name = "deepset/roberta-base-squad2"
fb_ai = pipeline('question-answering', model=model_name, tokenizer=model_name)

