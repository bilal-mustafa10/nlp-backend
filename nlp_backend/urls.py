"""nlp_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

import nlp_backend.views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('snack/', nlp_backend.views.return_highest_snack_country, name='Get highest snack country'),
    path('question/', nlp_backend.views.q_a_facebook, name='Get Question and Answer'),
    path('snack/sentiment/', nlp_backend.views.sentiment_year_graph, name='Get Sentiment of a snack')
]

