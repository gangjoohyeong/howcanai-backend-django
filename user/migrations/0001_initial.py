# Generated by Django 4.2.3 on 2023-08-10 02:47

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('username', models.CharField(max_length=15, unique=True)),
                ('password', models.CharField(max_length=20)),
                ('email', models.EmailField(max_length=50)),
            ],
        ),
    ]