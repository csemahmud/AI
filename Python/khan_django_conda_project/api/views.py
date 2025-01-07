from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import render
from database.models import User
from .serializers import UserSerializer


@api_view(['GET'])
def getUsers(request):
    users = User.objects.all();
    serializer = UserSerializer(users, many=True)
    return Response(serializer.data)

@api_view(['POST'])
def addUser(request):
    serializer = UserSerializer(data = request.data)
    if serializer.is_valid():
        serializer.save()
    print(serializer.data)
    response = Response(serializer.data)
    message = str(response.data) + " has been SAVED successfully"
    return Response(message)

def viewUsers(request):
    users = User.objects.all();
    serializer = UserSerializer(users, many=True)
    return render(request, 'view_users.html', {'users': serializer.data})