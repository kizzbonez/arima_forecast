from django.urls import path
from .views import train_data,refresh_sales_data,predict_sales_with_model,training_status,train_data_manual

urlpatterns = [
    path('train/', train_data, name='train_data'),
    path('train-manual/', train_data_manual, name='train_data_manual'),
    path('training-status/', training_status, name='train_status'),
    path('refresh-sales/', refresh_sales_data, name="refresh_sales"),
    path('predict/', predict_sales_with_model, name="predict")  ,
]
