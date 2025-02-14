# Generated by Django 5.1.3 on 2025-01-07 02:27

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="User",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=100, unique=True)),
                ("email", models.CharField(max_length=100, unique=True)),
                ("domain", models.CharField(max_length=255)),
                ("age", models.IntegerField()),
                ("experience", models.IntegerField()),
                ("salary", models.FloatField()),
            ],
        ),
    ]
