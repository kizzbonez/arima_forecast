from django.urls import path
from .views import train_data,refresh_sales_data,predict_sales_with_model,training_status

urlpatterns = [
    path('train/', train_data, name='train_data'),
    path('training-status/', training_status, name='train_status'),
    path('refresh-sales/', refresh_sales_data, name="refresh_sales"),
    path('predict/', predict_sales_with_model, name="predict")  ,
]
