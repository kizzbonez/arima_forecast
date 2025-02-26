from django.db import models

class SalesData(models.Model):
    date = models.DateField()
    sold_qty = models.IntegerField(default=0)
    item_id = models.TextField(max_length=10,null=True,blank=True)

    def __str__(self):
        return f"{self.item_name} - {self.date}"
