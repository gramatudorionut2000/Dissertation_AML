from django.urls import path
from . import views

urlpatterns = [
    path('transactions/', views.transactions, name='transactions'),
    path('transactions/upload/', views.transaction_upload, name='transaction_upload'),
    path('transactions/create/', views.transaction_create, name='transaction_create'),
    path('transactions/<int:transaction_id>/', views.transaction_detail, name='transaction_detail'),
    path('transactions/<int:transaction_id>/update/', views.transaction_update, name='transaction_update'),
    path('transactions/<int:transaction_id>/delete/', views.transaction_delete, name='transaction_delete'),
    path('transactions/<int:transaction_id>/graph-data/', views.transaction_graph_data, name='transaction_graph_data'),
    path('transactions/process-for-inference/', views.process_for_inference, name='process_for_inference'),
    path('transactions/processing-status/', views.get_processing_status, name='processing_status'),
    path('models/inference/', views.inference_view, name='inference'),
    path('models/run-inference/', views.run_inference, name='run_inference'),
    path('models/training/', views.training_view, name='training'),
    
    path('models/training/status/<str:model_name>/', views.model_training_status, name='model_training_status'),
    path('models/training/status/<str:model_name>/api/', views.model_training_status_api, name='model_training_status_api'),
    path('models/explainability/', views.explainability_view, name='explainability'),
    path('models/data-status/', views.process_data_status, name='data_status_api'),
]