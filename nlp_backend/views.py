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
        snack_df.sort_values(by='doc_date', inplace=True)
        snack_df_by_month = snack_df.groupby(pd.PeriodIndex(snack_df['doc_date'], freq="M"))[
            'doc_sentiment'].mean().reset_index()

        snack_df_by_month['doc_date'] = snack_df_by_month['doc_date'].astype(str)
        snack_df_by_month['doc_date'] = pd.to_datetime(snack_df_by_month['doc_date'])
        snack_df_by_month['doc_date'] = snack_df_by_month['doc_date'].dt.strftime('%Y-%m')
        snack_df_by_month['doc_sentiment'] = snack_df_by_month['doc_sentiment']


        #snack_df['doc_date'] = snack_df_by_month['doc_date'].dt.year
        # Smooth the sentiment data
        snack_df['doc_sentiment'] = snack_df_by_month['doc_sentiment']
        snack_df['doc_sentiment'] = snack_df_by_month['doc_sentiment'].fillna('')

        # return list of dictionaries with date and sentiment as x and y
        return HttpResponse(json.dumps(snack_df_by_month.to_dict('records')), content_type='application/json')


@csrf_exempt
def snack_continent_pie_chart(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        snack = data["snack"]
        continent_dict = get_continent_dist_for_a_snack(snack)
        return HttpResponse(json.dumps(continent_dict), content_type='application/json')
        

# Find highest talked of country
def find_snack_highest_talked_country(snack):
    countries = {}
    for country in df['doc_publish_location'].unique():
        countries[country] = df[df['doc_publish_location'] == country]['sentence'].str.contains(snack).sum()

    # Find the country that has the highest count
    return max(countries, key=countries.get)


def get_continent_dist_for_a_snack(snack):
    valuecounting = df.doc_publish_location.value_counts().to_dict()
    countries=[]
    for key,value in valuecounting.items():
    if ',' in key:
        country = key.split(",",1)[1]
        countries.append(country)
        df["doc_publish_location"]=df["doc_publish_location"].replace([key], country, regex=True)
    north_america=[" United States", " Canada", "Mexico"]
    north_america_total = 0
    south_america=[" Brazil", ' Columbia', ' Venuzuela', ' Argentina', ' Jamaica']
    south_america_total = 0
    europe=[' United Kingdom', ' France', ' Ireland',' Germany', ' Spain', ' Montenegro',' European Union', ' Cyprus', ' Austria', ' Netherlands', ' Croatia', ' Russia', ' Switzerland', ' Denmark', ' Malta', ' Latvia', ' Hungary', ' Luxembourg', ' Azerbaijan',' Greece', ' Ukraine', ' Romania', ' Gibraltar', ' Italy', ' Georgia']
    europe_total = 0
    asia_total=0
    africa=[" South Africa", ' Nigeria', ' Kenya', ' Ghana', ' Zimbabwe', ' Trinidad and Tobago', ' Rwanda', ' Uganda', ' Tanzania']
    africa_total = 0
    for i,j in valuecounting.items():
    if i in north_america:
        north_america_total += j
    elif i in south_america:
        south_america_total += j
    elif i in europe:
        europe_total += j
    elif i in africa:
        africa_total += j
    else:
        asia_total += j

    continent_dict = {"North America": north_america_total, "South America":south_america_total, "Asia": asia_total, "Africa": africa_total, "Europe":europe_total}
    
    return continent_dict