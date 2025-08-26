import json

from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required

from .forms import RegisterForm, JournalEntryForm
from .models import JournalEntry
from .ai_utils import predict_emotions

def home_view(request):
    return render(request, 'journal/home.html')

def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = RegisterForm()
    return render(request, 'journal/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request,user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request,'journal/login.html',{'form':form})

def logout_view(request):
    logout(request)
    return redirect('home')

@login_required
def journal_home(request):
    entries = JournalEntry.objects.filter(user = request.user).order_by('-created_at')
    return  render(request,'journal/journal_home.html',{'entries':entries})

@login_required
def journal_create(request):
    if request.method == 'POST':
        form = JournalEntryForm(request.POST)
        if form.is_valid():
            journal_entry = form.save(commit=False)
            journal_entry.user = request.user
            emotions = predict_emotions(journal_entry.content)

            journal_entry.sentiment = emotions[0][0]
            journal_entry.score = emotions[0][1]
            
            # Save all emotions as JSON for detailed analysis
            emotions_dict = {emotion: round(score, 1) for emotion, score in emotions}
            journal_entry.emotions_json = json.dumps(emotions_dict)

            request.session['all_emotions'] = emotions

            journal_entry.save()
            return redirect('journal_home')
    else:
        form = JournalEntryForm()
    return render(request,'journal/journal_create.html',{"form":form})

@login_required
def view_entry(request, entry_id):
    try:
        entry = JournalEntry.objects.get(id=entry_id, user=request.user)
    except JournalEntry.DoesNotExist:
        return redirect('journal_home')
    
    if request.method == 'POST':
        content = request.POST.get('content')
        if content:
            entry.content = content
            emotions = predict_emotions(entry.content)

            entry.sentiment = emotions[0][0]
            entry.score = emotions[0][1]
            
            # Save all emotions as JSON for detailed analysis
            emotions_dict = {emotion: round(score, 1) for emotion, score in emotions}
            entry.emotions_json = json.dumps(emotions_dict)

            request.session['all_emotions'] = emotions
            entry.save()
            return redirect('journal_home')
    
    return render(request, 'journal/view_entry.html', {'entry': entry})

@login_required
def delete_entry(request, entry_id):
    try:
        entry = JournalEntry.objects.get(id=entry_id, user=request.user)
        if request.method == 'POST':
            entry.delete()
            return redirect('journal_home')
        return render(request, 'journal/delete_confirm.html', {'entry': entry})
    except JournalEntry.DoesNotExist:
        return redirect('journal_home')