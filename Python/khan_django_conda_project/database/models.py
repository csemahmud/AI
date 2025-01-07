from django.db import models

# Create your models here.

class User(models.Model):
   id = models.AutoField(primary_key=True)  # Primary Key, Auto Increment
   name = models.CharField(max_length=100, unique=True)  # String, Unique
   email = models.CharField(max_length=100, unique=True)  # String, Unique
   domain = models.CharField(max_length=255)  # String (default max length can be adjusted)
   age = models.IntegerField()  # Integer
   experience = models.IntegerField()  # Integer
   salary = models.FloatField()  # Double/Float
   
   def __str__(self):
       return self.name
