from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('symptoms/',views.symptoms_model , name = "symptoms" ),
]
