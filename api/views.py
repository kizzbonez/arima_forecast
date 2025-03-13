import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.arima.model import ARIMA
from django.http import JsonResponse
from rest_framework.decorators import api_view
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .models import SalesData,TrainingStatus
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import pickle
from itertools import product
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
import subprocess
logger = logging.getLogger(__name__)
THIRD_PARTY_API_URL = "http://server.vade.dev:7877/v1/salesorderForArima"  # Waiting for sales API

# Function to calculate SMAPE
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

@api_view(['GET'])
def train_data_manual(request):
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

    # Possible ARIMA orders to try
    p_values = range(0, 6)
    d_values = range(0, 2)
    q_values = range(0, 6)
    orders = list(product(p_values, d_values, q_values))  # Generate all combinations

    for item in df['item_id'].unique():
        item_df = df[df['item_id'] == item][['date', 'sold_qty']].copy()
        
        # Set date as index
        item_df.set_index('date', inplace=True)

        # Resample to weekly frequency
        item_df = item_df.resample('W').sum()

        # Apply rolling mean for smoothing (4-week window)
        item_df['sold_qty'] = item_df['sold_qty'].rolling(window=4, min_periods=1).mean()

        # Ensure at least 12 data points (12 weeks)
        if len(item_df) < 12:
            predictions[item] = "Not enough data for ARIMA prediction"
            continue

        # Scale data to improve ARIMA stability
        scaler = StandardScaler()
        item_df_scaled = pd.DataFrame(
            scaler.fit_transform(item_df), index=item_df.index, columns=['sold_qty']
        )

        # Split data: 80% train, 20% test
        train_size = int(len(item_df_scaled) * 0.8)
        train_data = item_df_scaled.iloc[:train_size]
        test_data = item_df_scaled.iloc[train_size:]

        best_model = None
        best_order = None
        best_smape = float('inf')
        best_forecast = None

        try:
            # Iterate over all possible ARIMA orders
            warnings.simplefilter("ignore", ConvergenceWarning)
            for order in orders:
                try:
                    # Train ARIMA model
                    model = ARIMA(train_data, order=order)
                    model_fit = model.fit(method="innovations_mle")

                    # Forecast test data
                    forecast_scaled = model_fit.forecast(steps=len(test_data))
                    forecast = scaler.inverse_transform(forecast_scaled.values.reshape(-1, 1)).flatten()

                    # Convert test data back to original scale
                    test_actual = scaler.inverse_transform(test_data.values.reshape(-1, 1)).flatten()

                    # Calculate SMAPE
                    smape_value = smape(test_actual, forecast)

                    # If SMAPE ≤ 30 and it's the best found so far, update the best model
                    if smape_value <= 30 and smape_value < best_smape:
                        best_smape = smape_value
                        best_model = model_fit
                        best_order = order
                        best_forecast = forecast

                except Exception as e:
                    continue  # Ignore orders that fail to fit

            if best_model is not None:
                # Save the best model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"{models_dir}/arima_best_weekly_model_{item}_{timestamp}.pkl"

                with open(model_filename, 'wb') as model_file:
                    pickle.dump(best_model, model_file)

                # Predict for next 10 weeks
                future_dates = pd.date_range(start=item_df.index[-1], periods=10, freq='W')
                future_forecast_scaled = best_model.forecast(steps=10)
                future_forecast = scaler.inverse_transform(future_forecast_scaled.values.reshape(-1, 1)).flatten()

                # Calculate additional accuracy metrics
                mae = mean_absolute_error(test_actual, best_forecast)
                mse = mean_squared_error(test_actual, best_forecast)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((test_actual - best_forecast) / test_actual)) * 100

                predictions[item] = {
                    "best_order": best_order,
                    "forecast": dict(zip(future_dates.strftime('%Y-%m-%d'), future_forecast.tolist())),
                    "accuracy_metrics": {
                        "MAE": round(mae, 2),
                        "RMSE": round(rmse, 2),
                        "MAPE": round(mape, 2),
                        "SMAPE": round(best_smape, 2)
                    },
                    "model_saved": model_filename
                }
            else:
                predictions[item] = "No suitable ARIMA model found with SMAPE ≤ 30"

        except Exception as e:
            predictions[item] = str(e)

    return JsonResponse(predictions)


@api_view(['GET'])
def train_data(request):
    # Update training status to PENDING
    TrainingStatus.objects.all().delete()  # Keep only the latest status
    status = TrainingStatus.objects.create(status="PENDING")

    # Trigger background training process
    subprocess.Popen(["python", "manage.py", "run_training"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return JsonResponse({"message": "Training started", "status": "PENDING"})
    
@api_view(['GET'])
def training_status(request):
    status = TrainingStatus.objects.first()
    if status:
        return JsonResponse({
            "status": status.status,
            "last_run": status.last_run.strftime("%Y-%m-%d %H:%M:%S"),
            "error_message": status.error_message or ""
        })
    else:
        return JsonResponse({"status": "UNKNOWN"})

@api_view(['GET'])
def predict_sales_with_model(request):
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

    if not os.path.exists(models_dir):
        return JsonResponse({"error": "No models directory found"}, status=400)

    predictions = {}

    # Find saved models
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

    if not model_files:
        return JsonResponse({"error": "No saved models found"}, status=400)

    # Possible ARIMA orders to try


    for item in df['item_id'].unique():
         # Find the most recent model file for the item_id
        model_files_for_item = [f for f in model_files if f"_{item}_" in f]
        if not model_files_for_item:
            predictions[item] = "No saved model found for this item."
            continue
        
        model_files_for_item.sort(reverse=True)  # Sort to get the latest model
        model_file = model_files_for_item[0]
        model_path = os.path.join(models_dir, model_file)
        
        # Load the ARIMA model and scaler
        with open(model_path, 'rb') as file:
            loaded_data = pickle.load(file)
            if isinstance(loaded_data, tuple):
                model, scaler = loaded_data
            else:
                model = loaded_data
                scaler = StandardScaler()
        item_df = df[df['item_id'] == item][['date', 'sold_qty']].copy()
        
        # Set date as index
        item_df.set_index('date', inplace=True)

        # Resample to weekly frequency
        item_df = item_df.resample('W').sum()

        # Apply rolling mean for smoothing (4-week window)
        item_df['sold_qty'] = item_df['sold_qty'].rolling(window=4, min_periods=1).mean()

        # Ensure at least 12 data points (12 weeks)
        if len(item_df) < 12:
            predictions[item] = "Not enough data for ARIMA prediction"
            continue

        # Scale data to improve ARIMA stability
        item_df_scaled = pd.DataFrame(
            scaler.fit_transform(item_df), index=item_df.index, columns=['sold_qty']
        )

        # Split data: 80% train, 20% test
        train_size = int(len(item_df_scaled) * 0.8)
        train_data = item_df_scaled.iloc[:train_size]
        test_data = item_df_scaled.iloc[train_size:]


        # Forecast test data
        forecast_scaled = model.forecast(steps=len(test_data))
        forecast = scaler.inverse_transform(forecast_scaled.values.reshape(-1, 1)).flatten()

        # Convert test data back to original scale
        test_actual = scaler.inverse_transform(test_data.values.reshape(-1, 1)).flatten()

        # Calculate SMAPE
        smape_value = smape(test_actual, forecast)

        future_dates = pd.date_range(start=item_df.index[-1], periods=10, freq='W')
        future_forecast_scaled = model.forecast(steps=10)
        future_forecast = scaler.inverse_transform(future_forecast_scaled.values.reshape(-1, 1)).flatten()             
        # Calculate additional accuracy metrics
        mae = mean_absolute_error(test_actual, forecast)
        mse = mean_squared_error(test_actual, forecast)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_actual - forecast) / test_actual)) * 100

        predictions[item] = {
            "forecast": dict(zip(future_dates.strftime('%Y-%m-%d'), future_forecast.tolist())),
            "accuracy_metrics": {
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "MAPE": round(mape, 2),
                "SMAPE": round(smape_value, 2)
            }
        }
        

    return JsonResponse(predictions)



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

