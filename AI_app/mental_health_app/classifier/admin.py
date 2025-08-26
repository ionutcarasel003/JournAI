from django.contrib import admin

# Register your models here.

from django.contrib import admin
from .models import JournalEntry

@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'content', 'sentiment', 'score')
    search_fields = ('content', 'sentiment')
    list_filter = ('sentiment', 'created_at')
