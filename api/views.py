import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.arima.model import ARIMA
from django.http import JsonResponse
from rest_framework.decorators import api_view
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .models import SalesData
from datetime import datetime 
from django.db import transaction
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
import logging
import csv
import io
import joblib
from dateutil import parser
import itertools
from django.http import HttpRequest
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
logger = logging.getLogger(__name__)
THIRD_PARTY_API_URL = "http://server.vade.dev:7877/v1/salesorderForArima"  # Waiting for sales API


@api_view(['GET'])
def predict_sales(request):
    # Fetch sales data
    data = SalesData.objects.all().values('date', 'item_id', 'sold_qty')
    df = pd.DataFrame(data)

    if df.empty:
        return JsonResponse({"error": "No sales data available"}, status=400)

    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Group by date and item_id, then sum sold_qty
    df = df.groupby(['date', 'item_id']).agg({'sold_qty': 'sum'}).reset_index()

    # Ensure index is a proper DateTime index and sorted
    df = df.sort_values('date')

    predictions = {}

    # Ensure models directory exists
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for item in df['item_id'].unique():
        item_df = df[df['item_id'] == item][['date', 'sold_qty']].copy()
        
        # Set date as index
        item_df.set_index('date', inplace=True)

        # Fill missing dates with forward-fill
        item_df = item_df.asfreq('D').ffill()

        # Apply rolling mean for smoothing
        item_df['sold_qty'] = item_df['sold_qty'].rolling(window=7, min_periods=1).mean()

        # Ensure at least 12 data points
        if len(item_df) < 12:
            predictions[item] = "Not enough data for ARIMA prediction"
            continue

        # Scale data to improve ARIMA stability
        scaler = MinMaxScaler()
        item_df_scaled = pd.DataFrame(
            scaler.fit_transform(item_df), index=item_df.index, columns=['sold_qty']
        )

        # Split data: 80% train, 20% test
        train_size = int(len(item_df_scaled) * 0.8)
        train_data = item_df_scaled.iloc[:train_size]
        test_data = item_df_scaled.iloc[train_size:]

        try:
            # Train ARIMA model (fixing duplicate index issue)
            train_data = train_data.groupby(train_data.index).sum()

            model = ARIMA(train_data, order=(5,2,1))
            model_fit = model.fit()

            # Save the trained model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{models_dir}/arima_model_{item}_{timestamp}.pkl"

            with open(model_filename, 'wb') as model_file:
                pickle.dump(model_fit, model_file)

            # Forecast test data and scale back predictions
            forecast_scaled = model_fit.forecast(steps=len(test_data))
            forecast = scaler.inverse_transform(forecast_scaled.values.reshape(-1, 1)).flatten()

            # Convert test data back to original scale
            test_actual = scaler.inverse_transform(test_data.values.reshape(-1, 1)).flatten()

            # Calculate accuracy metrics
            mae = mean_absolute_error(test_actual, forecast)
            mse = mean_squared_error(test_actual, forecast)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test_actual - forecast) / test_actual)) * 100

            # Predict for next 30 days
            future_dates = pd.date_range(start=item_df.index[-1], periods=30, freq='D')
            future_forecast_scaled = model_fit.forecast(steps=30)
            future_forecast = scaler.inverse_transform(future_forecast_scaled.values.reshape(-1, 1)).flatten()

            predictions[item] = {
                "forecast": dict(zip(future_dates.strftime('%Y-%m-%d'), future_forecast.tolist())),
                "accuracy_metrics": {
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "MAPE": round(mape, 2)
                },
                "model_saved": model_filename
            }

        except Exception as e:
            predictions[item] = str(e)

    return JsonResponse(predictions)


# @api_view(['GET'])
# def predict_sales(request):
#     # Fetch sales data
#     data = SalesData.objects.all().values('date', 'item_id', 'sold_qty')
#     df = pd.DataFrame(data)
#     if df.empty:
#         return JsonResponse({"error": "No sales data available"}, status=400)

#     # Convert date column to datetime format and set as index
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.sort_values('date')
#     df.set_index('date', inplace=True)

#     predictions = {}

#     for item in df['item_id'].unique():
#         item_df = df[df['item_id'] == item][['sold_qty']]
        
#         # Convert irregular sales data into a daily time series
#         item_df = item_df.resample('D').mean().interpolate()
    
#         # Ensure at least 12 months of data for training/testing
#         if len(item_df) < 12:
#             predictions[item] = "Not enough data for ARIMA prediction"
#             continue

#         # Split data: 80% for training, 20% for testing
#         train_size = int(len(item_df) * 0.8)
#         train_data = item_df[:train_size]
#         test_data = item_df[train_size:]

#         try:
#             # Train ARIMA model
       
#             model = ARIMA(train_data, order=(2, 1, 0))
#             model_fit = model.fit()
            
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
#             # Define the filename with timestamp
#             filename = f"arima_model_{timestamp}.pkl"
            
#             # Save the model

#             # Forecast the same number of points as test_data
#             forecast = model_fit.forecast(steps=len(test_data))

#             # Calculate accuracy metrics
#             mae = mean_absolute_error(test_data, forecast)
#             mse = mean_squared_error(test_data, forecast)
#             rmse = np.sqrt(mse)
#             mape = np.mean(np.abs((test_data['sold_qty'] - forecast) / test_data['sold_qty'])) * 100

#             # Predict for next 30 days
#             future_dates = pd.date_range(start=item_df.index[-1], periods=30, freq='D')
#             future_forecast = model_fit.forecast(steps=30)

#             predictions[item] = {
#                 "forecast": dict(zip(future_dates.strftime('%Y-%m-%d'), future_forecast.tolist())),
#                 "accuracy_metrics": {
#                     "MAE": round(mae, 2),
#                     "RMSE": round(rmse, 2),
#                     "MAPE": round(mape, 2)
#                 }
#             }

#         except Exception as e:
#             predictions[item] = str(e)

#     return JsonResponse(predictions)



# @api_view(['POST'])
# def refresh_sales_data(request):
#     """Clears the sales table and fetches new sales data from a third-party API."""
#     try:
#         logger.info("Fetching sales data from third-party API...")

#         #  Step 1: Fetch new data from third-party API
#         response = requests.get(THIRD_PARTY_API_URL)
#         if response.status_code != 200:
#             logger.error(f"Failed to fetch sales data. Status Code: {response.status_code}")
#             return Response({"error": f"Failed to fetch data, API returned {response.status_code}"}, status=500)

#         new_sales_data = response.json()  # Convert response to JSON

#         #  Step 2: Validate response format
#         if not isinstance(new_sales_data, list):
#             logger.error(f"Invalid data format received: {new_sales_data}")
#             return Response({"error": "Invalid data format from API"}, status=400)

#         #  Step 3: Delete old sales records safely
#         with transaction.atomic():  # Ensures rollback if anything fails
#             logger.info("Clearing existing sales data...")
#             SalesData.objects.all().delete()

#             #  Step 4: Insert new sales data
#             sales_objects = []
#             for item in new_sales_data:
#                 try:
#                     # Validate date format
#                     sale_date = datetime.strptime(item["date"], "%Y-%m-%d").date()

#                     sales_objects.append(SalesData(
#                         date=sale_date,
#                         item_name=item["item_name"],
#                         sales=item["sales"],
#                         current_stocks=item["current_stocks"]
#                     ))

#                 except KeyError as e:
#                     logger.error(f"Missing required fields in data: {item}, Error: {str(e)}")
#                     return Response({"error": f"Missing required fields: {e}"}, status=400)
#                 except ValueError as e:
#                     logger.error(f"Invalid date format in data: {item}, Error: {str(e)}")
#                     return Response({"error": f"Invalid date format: {e}"}, status=400)

#             # Bulk insert new records
#             logger.info(f"Inserting {len(sales_objects)} new records into SalesData table...")
#             SalesData.objects.bulk_create(sales_objects)

#         logger.info("Sales data successfully refreshed.")
#         return Response({"message": "Sales data successfully refreshed"}, status=200)

#     except requests.RequestException as e:
#         logger.error(f"API request failed: {str(e)}", exc_info=True)
#         return Response({"error": "Failed to connect to third-party API"}, status=500)

#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         return Response({"error": f"Internal Server Error: {str(e)}"}, status=500)





@api_view(['POST'])
def refresh_sales_data(request: HttpRequest):
    """Clears the  sold item  table and fetches new  sold item  data from a CSV file or third-party API."""
    try:
        new_sales_data = []
        
        # Step 1: Check if a CSV file is provided
        if 'file' in request.FILES:
            logger.info("Processing uploaded CSV file...")
            csv_file = request.FILES['file']
            decoded_file = csv_file.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(decoded_file))
            for row in reader:
                try:
                    try:
                          # Attempt to parse the date and format it correctly
                        sale_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
                    except ValueError:
                        # If the format is incorrect, try auto-parsing it
                        sale_date = parser.parse(row["date"]).date()
                    new_sales_data.append({
                        "date": sale_date,
                        "item_id": row["item_id"] ,
                        "sold_qty": int(row["sold_qty"] or 0),
                    })
                except (KeyError, ValueError) as e:
                    logger.error(f"Invalid CSV data: {row}, Error: {str(e)}")
                    return Response({"error": f"Invalid CSV data format: {e}"}, status=400)
        else:
            # Step 2: Fetch new data from third-party API if CSV is not provided
            logger.info("Fetching sold item  data from third-party API...")
            response = requests.get(THIRD_PARTY_API_URL)
            if response.status_code != 200:
                logger.error(f"Failed to fetch  sold item data. Status Code: {response.status_code}")
                return Response({"error": f"Failed to fetch data, API returned {response.status_code}"}, status=500)
            
            sales_data = response.json()
            if not isinstance(sales_data, list):
                logger.error(f"Invalid data format received: {sales_data}")
                return Response({"error": "Invalid data format from API"}, status=400)

            for order in sales_data:
                sale_date = datetime.strptime(order["orderDate"], "%Y-%m-%d").strftime("%Y-%m-%d")

                for detail in order["salesorderDetailReportArimaDto"]:
                    new_sales_data.append({
                        "date": sale_date,
                        "item_id": detail["item"],  # Directly use "item" as it's an integer
                        "sold_qty": int(detail["orderQuantity"]),  # Convert quantity to int
                    })
                    
        with transaction.atomic():
            logger.info("Clearing existing sales data...")
            SalesData.objects.all().delete()
            
            sales_objects = [
                SalesData(
                    date=item["date"],
                    item_id=item["item_id"],
                    sold_qty=item["sold_qty"],
                ) for item in new_sales_data
            ]
          
            logger.info(f"Inserting {len(sales_objects)} new records into SalesData table...")
            SalesData.objects.bulk_create(sales_objects)
        
        logger.info("Sales data successfully refreshed.")
        return Response({"message": "Sales data successfully refreshed"}, status=200)
    
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}", exc_info=True)
        return Response({"error": "Failed to connect to third-party API"}, status=500)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return Response({"error": f"Internal Server Error: {str(e)}"}, status=500)

