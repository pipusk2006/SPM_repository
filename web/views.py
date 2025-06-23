from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Profile, InputData
from .forms import UserRegistrationForm, ProfileUpdateForm
from ML_models.model import RandomForestModel

# Импортируем модель только когда она нужна
def get_model_prediction(*args):
    return RandomForestModel(*args)

# Create your views here.

def home(request):
    return render(request, 'web/home.html')

def about(request):
    return render(request, 'web/about.html')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
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
def result(request, pk):
    try:
        input_data = InputData.objects.get(pk=pk, user=request.user)
        prob = input_data.prediction

        # Определяем уровень риска
        if prob <= 10:
            risk_level = "крайне низкий"
        elif prob <= 25:
            risk_level = "низкий"
        elif prob <= 50:
            risk_level = "значительный"
        elif prob <= 75:
            risk_level = "высокий"
        else:
            risk_level = "крайне высокий"

        context = {
            'input_data': input_data,
            'probability': prob,
            'risk_level': risk_level,
            'hypertension': input_data.hypertension,
            'smoking_status': input_data.smoking_status,
            'bmi': input_data.bmi,
            'age': input_data.age,
            'avg_glucose_level': input_data.avg_glucose_level
        }
        return render(request, 'web/result.html', context)

    except InputData.DoesNotExist:
        messages.error(request, 'Запись не найдена')
        return redirect('web:input_data')

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        messages.success(request, 'Вы успешно вышли из аккаунта!')
        return redirect('web:home')
    return render(request, 'web/logout.html')