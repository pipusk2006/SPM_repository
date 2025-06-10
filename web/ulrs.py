from django.urls import path
from . import views

app_name = 'web'

urlpatterns = [
    path('', views.index)
    path('about_us', views.about)
]