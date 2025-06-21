from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'web'

urlpatterns = [
    # Основные страницы
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('input_data/', views.input_data, name='input_data'),
    path('result/<int:pk>/', views.result, name='result'),
    
       # Аутентификация
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(
        template_name='web/login.html',
        redirect_authenticated_user=True
    ), name='login'),
    path('profile/', views.profile, name='profile'),
    path('logout/', auth_views.LogoutView.as_view(
        template_name='web/logout.html',
        next_page='web:home'
    ), name='logout'),
]