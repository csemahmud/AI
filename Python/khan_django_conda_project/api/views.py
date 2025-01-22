from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.shortcuts import render
from database.models import User
from .serializers import UserSerializer
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

@api_view(['GET'])
def getUserById(request, id):
   try:
       # Retrieve the user by ID
       user = User.objects.filter(id=id).first()
       if user:
           # Serialize the user object and return the response
           serializer = UserSerializer(user)
           return Response(serializer.data, status=200)
       else:
           # User not found
           return Response("User not found", status=404)
   except Exception as ex:
       # Handle unexpected errors
       return Response(str(ex), status=500)
   
@api_view(['GET'])
def getUserByEmail(request, email):
   try:
       # Retrieve the user by email
       user = User.objects.filter(email=email).first()
       if not user:
           return Response("User not found", status=404)
       # Serialize the user object and return it
       serializer = UserSerializer(user)
       return Response(serializer.data, status=200)
   except Exception as ex:
       # Handle unexpected errors
       return Response(str(ex), status=500)

@api_view(['PUT'])
def updateUser(request):
   try:
       # Find the user by ID
       user_id = request.data.get('id')  # Extract the user ID from the request data
       user = User.objects.filter(id=user_id).first()
       if not user:
           return Response("User not found", status=404)
       # Deserialize and validate the request data
       serializer = UserSerializer(instance=user, data=request.data, partial=True)  # Use partial=True to allow updating specific fields
       if serializer.is_valid():
           serializer.save()  # Save the updated user
           return Response(f"{serializer.data['name']} has been UPDATED successfully", status=200)
       return Response(serializer.errors, status=400)  # Validation errors
   except Exception as ex:
       return Response(str(ex), status=500)  # Handle unexpected errors
   
@api_view(['DELETE'])
def deleteUser(request, id):
   try:
       # Retrieve the user by ID
       user = User.objects.filter(id=id).first()
       if not user:
           return Response("User not found", status=404)
       # Delete the user
       user_name = user.name  # Capture name before deletion for response message
       user.delete()
       return Response(f"{user_name} has been DELETED successfully", status=200)
   except Exception as ex:
       # Handle unexpected errors
       return Response(str(ex), status=500)

def viewUsers(request):
    users = User.objects.all();
    serializer = UserSerializer(users, many=True)
    return render(request, 'view_users.html', {'users': serializer.data})

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