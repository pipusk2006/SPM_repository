from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from .models import Profile, InputData
from .forms import UserRegistrationForm, ProfileUpdateForm
from ML_models.model import RandomForestModel
from .models import InputData  # убедись, что модель есть
from django.contrib.auth.decorators import login_required
from django.contrib import messages

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



@login_required
def input_data(request):
    if request.method == 'POST':
        try:
            # Извлекаем данные из формы
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

            # Прогоняем через модель
            prediction = RandomForestModel(
                gender, age, hypertension, heart_disease,
                ever_married, work_type, residence_type,
                avg_glucose_level, bmi, smoking_status
            )

            # Сохраняем в БД
            input_instance = InputData.objects.create(
                user=request.user,
                gender=gender,
                age=age,
                hypertension=hypertension,
                heart_disease=heart_disease,
                ever_married=ever_married,
                work_type=work_type,
                Residence_type=residence_type,
                avg_glucose_level=avg_glucose_level,
                bmi=bmi,
                smoking_status=smoking_status,
                prediction=prediction
            )

            return redirect('web:result', pk=input_instance.pk)

        except Exception as e:
            messages.error(request, f'Ошибка при обработке данных: {e}')
            return redirect('web:input_data')

    # GET-запрос
    last_result = InputData.objects.filter(user=request.user).order_by('-id').first()
    return render(request, 'web/input_data.html', {'last_result': last_result})

