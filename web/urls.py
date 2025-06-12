from django.urls import path
from . import views

app_name = 'web'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('input_data/', views.input_data, name='input_data'),
    path('result/', views.result, name='result'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),  # Убедимся, что этот путь есть
    path('profile/', views.profile, name='profile'),
    path('logout/', views.logout_view, name='logout')  # Требуется представление logout_view
]