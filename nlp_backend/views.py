import json

from django.http import HttpResponse


def sethu_information(request):
    if request.method == 'GET':
        data = {
            "user": "Sethu",
            "date": "14-Feb-2001",
            "gender": "male"
        }
        return HttpResponse(json.dumps(data), content_type='application/json')

