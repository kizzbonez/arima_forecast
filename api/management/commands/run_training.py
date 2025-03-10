from django.core.management.base import BaseCommand
from .models import TrainingStatus
import logging
import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from django.db import transaction
from .models import SalesData

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Runs ARIMA training in the background"

    def handle(self, *args, **options):
        # Update status to RUNNING
        status = TrainingStatus.objects.first()
        status.status = "RUNNING"
        status.save()

        try:
            data = SalesData.objects.all().values('date', 'item_id', 'sold_qty')
            df = pd.DataFrame(data)

            if df.empty:
                status.status = "FAILED"
                status.error_message = "No sales data available"
                status.save()
                return

            df['date'] = pd.to_datetime(df['date'])
            df = df.groupby(['date', 'item_id']).agg({'sold_qty': 'sum'}).reset_index()
            df = df.sort_values('date')

            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)

            for item in df['item_id'].unique():
                item_df = df[df['item_id'] == item][['date', 'sold_qty']].copy()
                item_df.set_index('date', inplace=True)
                item_df = item_df.resample('W').sum()
                item_df['sold_qty'] = item_df['sold_qty'].rolling(window=4, min_periods=1).mean()

                if len(item_df) < 12:
                    continue

                scaler = StandardScaler()
                item_df_scaled = pd.DataFrame(scaler.fit_transform(item_df), index=item_df.index, columns=['sold_qty'])
                train_size = int(len(item_df_scaled) * 0.8)
                train_data = item_df_scaled.iloc[:train_size]

                model = ARIMA(train_data, order=(1, 1, 1))
                model_fit = model.fit()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"{models_dir}/arima_best_model_{item}_{timestamp}.pkl"

                with open(model_filename, 'wb') as model_file:
                    pickle.dump(model_fit, model_file)

            # Update status to SUCCESS
            status.status = "SUCCESS"
            status.error_message = ""
            status.save()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            status.status = "FAILED"
            status.error_message = str(e)
            status.save()
