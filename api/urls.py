from django.urls import path
from .views import predict_sales,refresh_sales_data

urlpatterns = [
    path('predict/', predict_sales, name='predict_sales'),
     path('refresh-sales/', refresh_sales_data, name="refresh_sales"),  # ✅ Add this API
]
