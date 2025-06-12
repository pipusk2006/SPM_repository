from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'accounts'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('input_data/', views.input_data, name='input_data'),
    path('result/', views.result, name='result'),  # Путь к странице с результатом
    path('register/', views.register, name='register'),  # Путь к странице регистрации
    path('login/', views.login, name='login'),  # Путь к странице входа
    path('profile/', views.profile, name='profile'),  # Путь к странице профиля
]
