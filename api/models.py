from django.db import models

class SalesData(models.Model):
    date = models.DateField()
    item_name = models.CharField(max_length=255)
    sales = models.IntegerField()
    current_stocks = models.IntegerField()

    def __str__(self):
        return f"{self.item_name} - {self.date}"
