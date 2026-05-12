from django.urls import path
from . import views

urlpatterns = [
    path('',                    views.index,            name='index'),
    path('login/',              views.login_page,        name='login'),
    path('register/',           views.register_page,     name='register'),
    path('logout/',             views.logout_view,       name='logout'),
    path('login/action/',       views.login_action,      name='login_action'),
    path('register/action/',    views.register_action,   name='register_action'),
    path('dashboard/',          views.dashboard,         name='dashboard'),
    path('otsu/',               views.otsu_page,          name='otsu'),
    path('otsu/run/',           views.otsu_action,        name='otsu_action'),
    path('dbim/',               views.dbim_page,          name='dbim'),
    path('dbim/run/',           views.dbim_action,        name='dbim_action'),
    path('accuracy/',           views.accuracy_page,      name='accuracy'),
    # AJAX validation endpoints
    path('api/check-username/', views.check_username,     name='check_username'),
    path('api/check-phone/',    views.check_phone,        name='check_phone'),
    path('api/check-email/',    views.check_email_ajax,   name='check_email'),
]
