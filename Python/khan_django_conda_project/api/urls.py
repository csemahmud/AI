from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.getUsers),
    path('addUser', views.addUser),
    path('addUser/', views.addUser),
    path('', views.viewUsers)
]