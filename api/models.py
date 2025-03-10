from django.db import models

class SalesData(models.Model):
    date = models.DateField()
    sold_qty = models.IntegerField(default=0)
    item_id = models.TextField(max_length=10,null=True,blank=True)

    def __str__(self):
        return f"{self.item_name} - {self.date}"


class TrainingStatus(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('SUCCESS', 'Success'),
        ('FAILED', 'Failed'),
    ]

    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    last_run = models.DateTimeField(auto_now=True)
    error_message = models.TextField(blank=True, null=True)

