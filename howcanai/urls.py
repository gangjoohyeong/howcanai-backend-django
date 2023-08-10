"""
URL configuration for howcanai project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from chatroom.api import ChatroomListView, ChatroomDetailView, QnaListView
from user.api import UserCreate, UserLogin

from django.conf import settings
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
    openapi.Info(
        title="Statchung API",
        default_version='v1',
        description="Test description",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@snippets.local"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)


urlpatterns = [
    path('admin/', admin.site.urls),
    
    
    path('api/chatroom/list', ChatroomListView.as_view(), name='chatroom_list'),
    path('api/chatroom/detail/<str:chatroom_id>', ChatroomDetailView.as_view(), name='chatroom_detail'),
    path('api/chatroom/update/<str:id>', ChatroomDetailView.as_view(), name='chatroom_update'),
    path('api/chatroom/delete/<str:id>', ChatroomDetailView.as_view(), name='chatroom_delete'),
    path('api/qna/create/<str:chatroom_id>', QnaListView.as_view(), name='qna_create'),
    
    path('api/user/create', UserCreate.as_view(), name='user_create'),
    path('api/user/login', UserLogin.as_view(), name='user_login'),
     
    path('swagger<str:format>', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
