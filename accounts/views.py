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

def home(request):
    return render(request, 'accounts/home.html')

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
