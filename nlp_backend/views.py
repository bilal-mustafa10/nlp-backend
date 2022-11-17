import json

from django.http import HttpResponse
from nlp_backend import interesting, fb_ai



def sethu_information(request):
    if request.method == 'GET':
        data = {
            "user": "Sethu",
            "date": "14-Feb-2001",
            "gender": "male"
        }
        return HttpResponse(json.dumps(data), content_type='application/json')



def return_highest_snack_country(request,snack):
    if request.method == 'GET':
        country = find_snack_highest_talked_country(snack)
        return HttpResponse(json.dumps(country), content_type='application/json')


# topic - "Walnuts" 
# question - "What are some famous walnut types?"
# answer - "Organic Walnuts and Organic Walnuts with Apple Cinnamon"
def q_a_facebook(request, topic, question):
    model_name = "deepset/roberta-base-squad2"
    fb_ai = pipeline('question-answering', model=model_name, tokenizer=model_name)
    sentences_topic = ' '.join(interesting[interesting['sentence'].str.contains(topic)]['sentence'])







# Find highest talked of country
def find_snack_highest_talked_country(snack):
  countries = {}
  for country in df['doc_publish_location'].unique():
      countries[country] = df[df['doc_publish_location'] == country]['sentence'].str.contains(snack).sum()

  # Find the country that has the highest count
  return max(countries, key=countries.get)

find_snack_highest_talked_country("pretzels")