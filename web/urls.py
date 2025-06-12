from django.urls import path
from . import views

app_name = 'web'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('input_data/', views.input_data, name='input_data'),
] 