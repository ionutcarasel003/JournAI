from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('journal/', views.journal_home, name='journal_home'),
    path('new/', views.journal_create, name='journal_create'),
    path('entry/<int:entry_id>/', views.view_entry, name='view_entry'),
    path('entry/<int:entry_id>/delete/', views.delete_entry, name='delete_entry'),
]
