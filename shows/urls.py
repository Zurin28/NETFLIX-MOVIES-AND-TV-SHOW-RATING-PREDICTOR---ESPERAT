from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),  # Changed next_page to 'login'
    path('predict/', views.predict_show, name='predict_show'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
]