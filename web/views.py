from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import numpy as np
from ML_models.model import RandomForestModel
from django.contrib import messages

# Create your views here.

def home(request):
    return render(request, 'web/home.html')

def about(request):
    return render(request, 'web/about.html')

def input_data(request):
    if request.method == "POST":
        try:
            # Проверяем наличие всех необходимых полей
            required_fields = ['gender', 'smoking_status', 'age', 'work_type', 
                             'Residence_type', 'hypertension', 'avg_glucose_level', 
                             'bmi', 'ever_married', 'heart_disease']
            
            for field in required_fields:
                if field not in request.POST:
                    raise KeyError(f'Поле {field} отсутствует в форме')

            # Получаем и преобразуем данные
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

            # Инициализация модели
            model = RandomForestModel()
            
            # Преобразование данных в числовой формат
            data = np.array([[age, hypertension, avg_glucose_level, bmi, heart_disease]])
            probability = model.predict_proba(data)[0][1] * 100  # Вероятность инсульта

            context = {
                'probability': round(probability, 2),
                'hypertension': hypertension,
                'smoking_status': smoking_status,
                'bmi': bmi,
                'age': age,
                'avg_glucose_level': avg_glucose_level,
            }
            return render(request, 'web/result.html', context)
        except KeyError as e:
            messages.error(request, f'Ошибка в форме: {str(e)}')
            return render(request, 'web/input_data.html')
        except ValueError as e:
            messages.error(request, 'Пожалуйста, проверьте правильность введенных числовых значений')
            return render(request, 'web/input_data.html')
        except Exception as e:
            messages.error(request, f'Произошла ошибка: {str(e)}')
            return render(request, 'web/input_data.html')
            
    return render(request, 'web/input_data.html')

def result(request):
    return render(request, 'web/result.html')

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
