from django.urls import path
from . import views


urlpatterns = [
    path('signup/', views.signup, name='signup'),
    path('login/',views.login_view,name='login'),
     path('fusion/', views.fusion_api, name='fusion_api'),

]