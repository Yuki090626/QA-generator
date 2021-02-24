from django.urls import path
from generator import views

app_name = 'generator'
urlpatterns = [
    path('', views.doc_register, name='doc_register'),   # 登録ページ
    path('result/', views.ResultList.as_view(), name='gen_result'),   # QA生成結果ページ
    path('del/<int:qac_idx>/', views.qac_del, name='qac_del'),   # QAの削除
    path('docs/', views.DocList.as_view(), name='doc_list'),     # ドキュメント一覧
    path('docs/qa/<int:doc_id>', views.doc_qa_list, name='doc_qa_list'),     # ドキュメント毎のQA一覧
]