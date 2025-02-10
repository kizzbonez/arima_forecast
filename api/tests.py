from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch
from rest_framework import status
from .models import SalesData
import json
import datetime
import logging

class SalesPredictionTest(TestCase):

    def setUp(self):
        """Set up initial sales data before each test."""
        SalesData.objects.create(date="2024-01-01", item_name="Product A", sales=100, current_stocks=50)
        SalesData.objects.create(date="2024-02-01", item_name="Product A", sales=120, current_stocks=40)
        SalesData.objects.create(date="2024-03-01", item_name="Product A", sales=140, current_stocks=30)

        SalesData.objects.create(date="2024-01-01", item_name="Product B", sales=80, current_stocks=70)
        SalesData.objects.create(date="2024-02-01", item_name="Product B", sales=90, current_stocks=60)
        SalesData.objects.create(date="2024-03-01", item_name="Product B", sales=110, current_stocks=50)

    def test_sales_prediction_api(self):
        """Test if API returns a valid response."""
        response = self.client.get(reverse('predict_sales'))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn("Product A", data)
        self.assertIn("Product B", data)

    def test_sales_prediction_with_no_data(self):
        """Test prediction when no data is present."""
        SalesData.objects.all().delete()
        response = self.client.get(reverse('predict_sales'))
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", json.loads(response.content))

    def test_sales_prediction_with_multiple_products(self):
        """Ensure the API correctly predicts for multiple products."""
        response = self.client.get(reverse('predict_sales'))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn("Product A", data)
        self.assertIn("Product B", data)
        self.assertIsInstance(data["Product A"], dict)
        self.assertIsInstance(data["Product B"], dict)

    def test_sales_prediction_with_insufficient_data(self):
        """Test if ARIMA handles insufficient historical data gracefully."""
        SalesData.objects.all().delete()
        SalesData.objects.create(date="2024-01-01", item_name="Product C", sales=50, current_stocks=20)
        
        response = self.client.get(reverse('predict_sales'))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["Product C"], "Not enough data for ARIMA prediction")
        self.assertTrue("not enough data" in str(data["Product C"]).lower() or isinstance(data["Product C"], dict))

    def test_sales_prediction_with_inconsistent_dates(self):
        """Test prediction when sales data has irregular dates."""
        SalesData.objects.create(date="2024-02-10", item_name="Product D", sales=60, current_stocks=30)
        SalesData.objects.create(date="2024-03-20", item_name="Product D", sales=80, current_stocks=20)

        response = self.client.get(reverse('predict_sales'))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn("Product D", data)
        self.assertIsInstance(data["Product D"], dict)

    def test_sales_prediction_with_large_dataset(self):
        """Test ARIMA prediction with a large dataset."""
        SalesData.objects.all().delete()
        base_date = datetime.date(2020, 1, 1)
        
        for i in range(1000):  # Add 1000 days of sales data
            SalesData.objects.create(date=base_date + datetime.timedelta(days=i),
                                     item_name="Product X",
                                     sales=100 + (i % 50),  # Simulated fluctuations
                                     current_stocks=500 - (i % 50))

        response = self.client.get(reverse('predict_sales'))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertIn("Product X", data)
        self.assertIsInstance(data["Product X"], dict)

class RefreshSalesDataTest(TestCase):
    """Test case for the refresh sales data API"""

    def setUp(self):
        """Set up initial sales data before each test."""
        SalesData.objects.create(date="2024-01-01", item_name="Product A", sales=100, current_stocks=50)
        SalesData.objects.create(date="2024-02-01", item_name="Product B", sales=200, current_stocks=40)

    @patch("requests.get")
    def test_refresh_sales_success(self, mock_get):
        """Test that sales data is deleted and new data is inserted successfully."""
        
        #  Mock API response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {"date": "2025-02-10", "item_name": "Laptop", "sales": 20, "current_stocks": 100},
            {"date": "2025-02-09", "item_name": "Mouse", "sales": 15, "current_stocks": 80}
        ]

        response = self.client.post(reverse('refresh_sales'))
        
        #  Log response to debug unexpected errors
        print("Test API Response:", response.content)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(SalesData.objects.count(), 2)  #  Updated
        self.assertTrue(SalesData.objects.filter(item_name="Laptop").exists())

    @patch("requests.get")
    def test_refresh_sales_api_failure(self, mock_get):
        """Test API failure when the third-party API is unreachable."""
        mock_get.return_value.status_code = 500  # Simulating API failure

        response = self.client.post(reverse('refresh_sales'))
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)

        #  Ensure old data is not deleted if the API fails
        self.assertEqual(SalesData.objects.count(), 2)  #  Updated

    @patch("requests.get")
    def test_refresh_sales_with_no_data(self, mock_get):
        """Test API response when no sales data is available."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = []  # Empty response

        response = self.client.post(reverse('refresh_sales'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(SalesData.objects.count(), 0)  #  Updated
