from django.urls import path
from . import views

urlpatterns = [
    path('canvas/', views.canvas, name='canvas'),
    path('submit-drawing/', views.submit_drawing, name='submit_drawing'),
    path('train-model/', views.train_model, name='train_model'),
    path('predict/', views.predict, name='predict'),
]
