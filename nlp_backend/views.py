import json

import pandas as pd
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline

from nlp_backend import interesting, fb_ai, df


@csrf_exempt
def return_highest_snack_country(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        snack = data["snack"]
        country = find_snack_highest_talked_country(snack)
        return HttpResponse(json.dumps(country), content_type='application/json')


@csrf_exempt
def q_a_facebook(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        question = data["question"]
        topic = data["topic"]
        model_name = "deepset/roberta-base-squad2"
        fb_ai = pipeline('question-answering', model=model_name, tokenizer=model_name)
        sentences_topic = ' '.join(interesting[interesting['sentence'].str.contains(topic)]['sentence'])

        if len(sentences_topic) > 25000:
            sentences_topic = sentences_topic[:10000]
        qa_input = {
            'question': question,
            'context': sentences_topic
        }
        result = fb_ai(qa_input)
        return HttpResponse(json.dumps(result), content_type='application/json')


@csrf_exempt
def sentiment_year_graph(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        snack = data["snack"]
        snack_df = df[df['sentence'].str.contains(snack)][['doc_date', 'doc_sentiment']]
        # convert doc_date to datetime
        snack_df['doc_date'] = pd.to_datetime(snack_df['doc_date'])
        # Smooth the sentiment data
        snack_df['doc_sentiment'] = snack_df['doc_sentiment'].fillna('')

        return_data = {
            "x": snack_df['doc_date'].dt.year.tolist(),
            "y": snack_df['doc_sentiment'].tolist()
        }
        # convert to a tuple of (year, sentiment)
        return_data = list(zip(return_data['x'], return_data['y']))
        return HttpResponse(json.dumps(return_data), content_type='application/json')


# Find highest talked of country
def find_snack_highest_talked_country(snack):
    countries = {}
    for country in df['doc_publish_location'].unique():
        countries[country] = df[df['doc_publish_location'] == country]['sentence'].str.contains(snack).sum()

    # Find the country that has the highest count
    return max(countries, key=countries.get)
