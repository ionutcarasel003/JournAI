import json

from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class JournalEntry(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    sentiment = models.CharField(max_length=50,blank=True,null=True)
    created_at = models.DateTimeField(auto_now=True)
    score = models.FloatField(blank=True,null=True)
    emotions_json = models.TextField(blank=True, null=True)

    def get_emotions_dict(self):
        try:
            return json.loads(self.emotions_json) if self.emotions_json else {}
        except:
            return {}

    def __str__(self):
        return f"{self.user.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
