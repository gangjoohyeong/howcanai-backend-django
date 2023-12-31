import json
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import *
from rest_framework import status, viewsets
from chat.run import run_chat



class ChatroomListView(APIView):
    permission_classes = [IsAuthenticated] 
    
    # /api/chatroom/list
    def get(self, request):
        user_chatrooms = Chatroom.objects.filter(user=request.user)
        serializer = ChatroomSerializer(user_chatrooms, many=True)
        return Response(serializer.data)

class ChatroomCreateView(APIView):
    permission_classes = [IsAuthenticated] 
    
    # /api/chatroom/create
    def post(self, request):
        data = request.data.copy()  # 입력 데이터 복사
        data['user'] = request.user.id  # 현재 로그인된 사용자의 ID를 user 필드에 추가
        
        serializer = ChatroomSerializer(data=data)
        if serializer.is_valid():
            chatroom = serializer.save()
            return Response({'chatroom_id': chatroom.id}, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatroomDetailView(APIView):
    permission_classes = [IsAuthenticated] 
    
    # /api/chatroom/detail/{chatroom_id}
    def get(self, request, chatroom_id):
        try:
            chatroom = Chatroom.objects.get(id=chatroom_id, user=request.user)
        except Chatroom.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        qnas = Qna.objects.filter(chatroom=chatroom)
        qna_serializer = QnaSerializer(qnas, many=True)
        chatroom_data = ChatroomSerializer(chatroom).data
        chatroom_data['qnas'] = qna_serializer.data
        return Response(chatroom_data)

    # /api/chatroom/update
    def put(self, request, id):
        chatroom = Chatroom.objects.get(id=id)
        serializer = ChatroomSerializer(chatroom, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors,status=status.HTTP_400_BAD_REQUEST)
    
    # /api/chatroom/delete
    def delete(self, request, id):
        try:
            chatroom = Chatroom.objects.get(id=id, user=request.user)
        except Chatroom.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        
        chatroom.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class QnaListView(APIView):
    permission_classes = [IsAuthenticated]

    # /api/qna/create/{chatroom_id}
    def post(self, request, chatroom_id):
        
        try:
            chatroom = Chatroom.objects.get(id=chatroom_id, user=request.user)
        except Chatroom.DoesNotExist:
            return Response({'detail': 'Chatroom not found.'}, status=status.HTTP_404_NOT_FOUND)
        query = request.data.get('query')
        answer, references, nexts = run_chat(query)
        qna_data = {
            'question': query,
            'answer': answer,
            'references': references,
            'nexts': json.dumps(nexts),
            'chatroom': chatroom.id  # 현재 chatroom과 연결
        }
        qna_serializer = QnaSerializer(data=qna_data)
        if qna_serializer.is_valid():
            qna_serializer.save()
            return Response(qna_serializer.data, status=status.HTTP_201_CREATED)
        return Response(qna_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
