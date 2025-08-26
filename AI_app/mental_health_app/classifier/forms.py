from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import JournalEntry

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class JournalEntryForm(forms.ModelForm):
    class Meta:
        model = JournalEntry
        fields = ['content']