from rest_framework import serializers
from .models import Chatroom, Qna

class QnaSerializer(serializers.ModelSerializer):    
    class Meta:
        model = Qna
        fields = '__all__'

class ChatroomSerializer(serializers.ModelSerializer):  
    class Meta:
        model = Chatroom
        fields = '__all__'
    
    def create(self, validated_data):
        if 'title' not in validated_data:
            validated_data['title'] = 'New Chat'
        return Chatroom.objects.create(**validated_data)
        
        
