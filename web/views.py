from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# Create your views here.

def home(request):
    return render(request, 'web/home.html')

def about(request):
    return render(request, 'web/about.html')

@login_required
def input_data(request):
    return render(request, 'web/input_data.html')
