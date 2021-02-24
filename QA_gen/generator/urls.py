from django.urls import path
from generator import views

app_name = 'generator'
urlpatterns = [
    path('', views.doc_register, name='doc_register'),   # 登録ページ
    path('result/', views.ResultList.as_view(), name='gen_result'),   # 結果ページ
]