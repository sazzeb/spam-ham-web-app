from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('multiple', views.multiple, name='multiple'),
    path('multiple_upload', views.multiple_upload, name='multiple_upload'),
]