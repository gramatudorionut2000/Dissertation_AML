from django.db import models
from django.utils.translation import gettext_lazy as _
from django.conf import settings

class Transaction(models.Model):

    timestamp = models.DateTimeField()
    
    from_bank = models.BigIntegerField()
    from_account = models.CharField(max_length=50)
    to_bank = models.BigIntegerField()
    to_account = models.CharField(max_length=50)
    
    amount_received = models.DecimalField(max_digits=20, decimal_places=2)
    receiving_currency = models.CharField(max_length=50)
    amount_paid = models.DecimalField(max_digits=20, decimal_places=2)
    payment_currency = models.CharField(max_length=50)
    
    payment_format = models.CharField(max_length=50)
    
    is_laundering = models.BooleanField()
    
    processed = models.BooleanField(default=False)

    date_added = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['from_bank']),
            models.Index(fields=['to_bank']),
            models.Index(fields=['receiving_currency']),
            models.Index(fields=['payment_currency']),
            models.Index(fields=['payment_format']),
            models.Index(fields=['is_laundering']),
            models.Index(fields=['processed']),
        ]
    
    def __str__(self):
        return f"Transaction {self.id} - {self.timestamp}"

