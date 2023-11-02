from django.contrib import admin
from django.urls import path

from . import views

urlpatterns = [
    path('', views.LeastSquaresView.as_view(), name='least_squares'),
    path('tangent_two_circles/', views.TangentTwoCirclesView.as_view(), name='tangent_two_circles'),
    path('polygon/', views.PolygonView.as_view(), name='polygon'),
]
