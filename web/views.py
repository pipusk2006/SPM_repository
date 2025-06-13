from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Profile, InputData
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent  
sys.path.append(str(project_root))
from ML_models.model import RandomForestModel

# Create your views here.

def home(request):
    return render(request, 'web/home.html')

def about(request):
    return render(request, 'web/about.html')

class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['date_of_birth', 'phone_number']

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Profile creation is handled by the signal in models.py
            login(request, user)
            messages.success(request, 'Регистрация успешна!')
            return redirect('web:profile')
    else:
        form = UserRegistrationForm()
    return render(request, 'web/register.html', {'form': form})

@login_required
def profile(request):
    if request.method == 'POST':
        profile_form = ProfileUpdateForm(request.POST, instance=request.user.profile)
        if profile_form.is_valid():
            profile_form.save()
            messages.success(request, 'Профиль успешно обновлен!')
            return redirect('web:profile')
    else:
        profile_form = ProfileUpdateForm(instance=request.user.profile)
    
    context = {
        'profile_form': profile_form
    }
    return render(request, 'web/profile.html', context)

@login_required
def input_data(request):
    if request.method == 'POST':
        try:
            # Получаем данные из формы
            gender = request.POST.get('gender')
            age = float(request.POST.get('age'))
            hypertension = int(request.POST.get('hypertension'))
            heart_disease = int(request.POST.get('heart_disease'))
            ever_married = request.POST.get('ever_married')
            work_type = request.POST.get('work_type')
            Residence_type = request.POST.get('Residence_type')
            avg_glucose_level = float(request.POST.get('avg_glucose_level'))
            bmi = float(request.POST.get('bmi'))
            smoking_status = request.POST.get('smoking_status')

            # Создаем запись в базе данных
            input_data = InputData.objects.create(
                user=request.user,
                gender=gender,
                age=age,
                hypertension=hypertension,
                heart_disease=heart_disease,
                ever_married=ever_married,
                work_type=work_type,
                Residence_type=Residence_type,
                avg_glucose_level=avg_glucose_level,
                bmi=bmi,
                smoking_status=smoking_status
            )

            # Получаем предсказание от модели
            prediction = RandomForestModel(
                gender, age, hypertension, heart_disease, ever_married,
                work_type, Residence_type, avg_glucose_level, bmi, smoking_status
            )

            # Сохраняем результат
            input_data.prediction = prediction
            input_data.save()

            return redirect('web:result', pk=input_data.pk)
        except Exception as e:
            messages.error(request, f'Произошла ошибка: {str(e)}')
            return redirect('web:input_data')
    return render(request, 'web/input_data.html')

@login_required
def result(request, pk):
    try:
        input_data = InputData.objects.get(pk=pk, user=request.user)
        context = {
            'input_data': input_data,
            'prediction': input_data.prediction
        }
        return render(request, 'web/result.html', context)
    except InputData.DoesNotExist:
        messages.error(request, 'Запись не найдена')
        return redirect('web:input_data')

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
