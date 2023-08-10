from django.db import models
import uuid

class User(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    username = models.CharField(max_length=15, unique=True, null=False)
    password = models.CharField(max_length=20, null=False)
    email = models.EmailField(max_length=50, null=False)