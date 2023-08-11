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
        
        
