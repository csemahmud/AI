from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.getUsers),
    path('addUser', views.addUser),
    path('addUser/', views.addUser),
    path('', views.viewUsers),
    path('user/<int:id>', views.getUserById, name='get-user-by-id'),
    path('user/<int:id>/', views.getUserById, name='get-user-by-id'),
    path('userByEmail/<str:email>', views.getUserByEmail, name='get-user-by-email'),  # Correct URL for getUserByEmail
    path('userByEmail/<str:email>/', views.getUserByEmail, name='get-user-by-email'),  # Correct URL for getUserByEmail
    path('update', views.updateUser, name='update-user'),
    path('update/', views.updateUser, name='update-user'),
    path('delete/<int:id>', views.deleteUser, name='delete-user'),  # Correct URL for deleteUser
    path('delete/<int:id>/', views.deleteUser, name='delete-user')  # Correct URL for deleteUser
]