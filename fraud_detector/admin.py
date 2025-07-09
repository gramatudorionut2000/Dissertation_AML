from django.contrib import admin
from .models import Transaction

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'timestamp', 'from_account', 'to_account', 
        'amount_received', 'receiving_currency', 
        'payment_format', 'is_laundering'
    )
    list_filter = (
        'is_laundering', 'receiving_currency', 
        'payment_currency', 'payment_format'
    )
    search_fields = (
        'from_account', 'to_account', 'from_bank', 'to_bank'
    )
    date_hierarchy = 'timestamp'
    readonly_fields = ('date_added', 'last_modified')
    
    fieldsets = (
        ('Temporal Info', {
            'fields': ('timestamp', 'date_added', 'last_modified')
        }),
        ('Accounts', {
            'fields': ('from_bank', 'from_account', 'to_bank', 'to_account')
        }),
        ('Financials', {
            'fields': (
                'amount_received', 'receiving_currency',
                'amount_paid', 'payment_currency',
                'payment_format'
            )
        }),
        ('Risk assessment', {
            'fields': ('is_laundering',),
            'classes': ('wide',)
        }),
    )