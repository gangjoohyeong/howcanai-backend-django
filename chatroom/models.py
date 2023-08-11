from django.db import models
from django.contrib.auth.models import User
import uuid

class Chatroom(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=30, null=False)
    create_date = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=False)


class Qna(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.TextField(null=False)
    answer = models.TextField(null=False)
    create_date = models.DateTimeField(auto_now_add=True)
    chatroom = models.ForeignKey(Chatroom, on_delete=models.CASCADE, null=False)
    nexts = models.TextField()