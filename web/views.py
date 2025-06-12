from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# Create your views here.

def home(request):
    return render(request, 'web/home.html')

def about(request):
    return render(request, 'web/about.html')

def input_data(request):
    return render(request, 'web/input_data.html')

def login(request):
    return render(request, 'web/login.html')



from django.shortcuts import render
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Пример модели (замени на реальную обученную модель)
model = RandomForestClassifier()
# Здесь должна быть загрузка обученной модели, например, из файла

def input_data(request):
    if request.method == "POST":
        gender = request.POST['gender']
        smoking_status = request.POST['smoking_status']
        age = float(request.POST['age'])
        work_type = request.POST['work_type']
        residence_type = request.POST['Residence_type']
        hypertension = int(request.POST['hypertension'])
        avg_glucose_level = float(request.POST['avg_glucose_level'])
        bmi = float(request.POST['bmi'])
        ever_married = request.POST['ever_married']
        heart_disease = int(request.POST['heart_disease'])

        # Преобразование данных в числовой формат (упрощенный пример)
        data = np.array([[age, hypertension, avg_glucose_level, bmi, heart_disease]])
        probability = model.predict_proba(data)[0][1] * 100  # Вероятность класса 1 (инсульт)

        context = {
            'probability': round(probability, 2),
            'hypertension': hypertension,
            'smoking_status': smoking_status,
            'bmi': bmi,
            'age': age,
            'avg_glucose_level': avg_glucose_level,
        }
        return render(request, 'result.html', context)
    return render(request, 'input_data.html')

def result(request):
    # Здесь можно передать те же данные, если переход через GET
    context = {
        'probability': 0,  # Замени на реальную логику
        'hypertension': 0,
        'smoking_status': 'never smoked',
        'bmi': 22.0,
        'age': 30,
        'avg_glucose_level': 90.0,
    }
    return render(request, 'result.html', context)
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

# Create your models here.

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    date_of_birth = models.DateField(null=True, blank=True)
    phone_number = models.CharField(max_length=15, blank=True)
    
    def __str__(self):
        return f'{self.user.username} Profile'

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegistrationForm, ProfileUpdateForm
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent  
sys.path.append(str(project_root))
from ML_models.model import RandomForestModel

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Profile creation is handled by the signal in models.py
            login(request, user)
            messages.success(request, 'Регистрация успешна!')
            return redirect('accounts:profile')
    else:
        form = UserRegistrationForm()
    return render(request, 'accounts/register.html', {'form': form})

@login_required
def profile(request):
    if request.method == 'POST':
        profile_form = ProfileUpdateForm(request.POST, instance=request.user.profile)
        if profile_form.is_valid():
            profile_form.save()
            messages.success(request, 'Профиль успешно обновлен!')
            return redirect('accounts:profile')
    else:
        profile_form = ProfileUpdateForm(instance=request.user.profile)
    
    context = {
        'profile_form': profile_form
    }
    return render(request, 'accounts/profile.html', context)
