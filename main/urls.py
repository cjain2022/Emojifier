from django.urls import path
from main import views
urlpatterns = [
    path('download_emb/',views.temp_view),
    path('',views.predictEmoji,name='predictEmoji'),
]