from rest_framework import serializers
from database.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'

    def update(self, instance, validated_data):
       # Update only specific fields
       for attr, value in validated_data.items():
           setattr(instance, attr, value)
       instance.save()
       return instance