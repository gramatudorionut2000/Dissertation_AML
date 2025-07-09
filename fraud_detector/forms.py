from django import forms

from django import forms

from .models import Transaction


class CSVUploadForm(forms.Form):

    csv_file = forms.FileField(
        label='Transaction CSV',
        help_text='Upload a CSV file with transaction data (Max size: 500MB)',
        widget=forms.FileInput(attrs={
            'style': 'display: none;',
            'accept': '.csv'
        })
    )
    
    has_header = forms.BooleanField(
        initial=True,
        required=False,
        label='File has header row',
        widget=forms.CheckboxInput(attrs={
            'style': 'width: 16px; height: 16px; border-radius: 4px; border: 1px solid #999;'
        })
    )
    
    delimiter = forms.ChoiceField(
        choices=[
            (',', 'Comma (,)'),
            (';', 'Semicolon (;)'),
            ('\t', 'Tab'),
            ('|', 'Pipe (|)'),
        ],
        initial=',',
        label='Field Delimiter',
        widget=forms.Select(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
        })
    )
    
    encoding = forms.ChoiceField(
        choices=[
            ('utf-8', 'UTF-8'),
            ('utf-8-sig', 'UTF-8 with BOM'),
            ('latin-1', 'Latin-1'),
            ('ascii', 'ASCII'),
        ],
        initial='utf-8',
        label='File Encoding',
        widget=forms.Select(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
        })
    )


class TransactionFilterForm(forms.Form):

    start_date = forms.DateField(
        required=False, 
        label='Start Date',
        widget=forms.DateInput(attrs={
            'type': 'date',
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
        })
    )
    
    end_date = forms.DateField(
        required=False, 
        label='End Date',
        widget=forms.DateInput(attrs={
            'type': 'date',
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
        })
    )
    
    receiving_currency = forms.CharField(
        required=False, 
        label='Receiving Currency',
        widget=forms.TextInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
            'placeholder': 'e.g., US Dollar, Euro'
        })
    )
    
    payment_currency = forms.CharField(
        required=False, 
        label='Payment Currency',
        widget=forms.TextInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
            'placeholder': 'e.g., US Dollar, Euro'
        })
    )
    
    payment_format = forms.CharField(
        required=False, 
        label='Payment Format',
        widget=forms.TextInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
            'placeholder': 'e.g., Cheque, Credit Card'
        })
    )
    
    min_amount = forms.DecimalField(
        required=False, 
        label='Minimum Amount',
        widget=forms.NumberInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
            'placeholder': 'Minimum amount'
        })
    )
    
    max_amount = forms.DecimalField(
        required=False, 
        label='Maximum Amount',
        widget=forms.NumberInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
            'placeholder': 'Maximum amount'
        })
    )
    
    fraud_only = forms.BooleanField(
        required=False, 
        label='Show Potential Fraud Only',
        widget=forms.CheckboxInput(attrs={
            'style': 'width: 16px; height: 16px; border-radius: 4px; border: 1px solid #999;'
        })
    )


class TransactionCreateForm(forms.ModelForm):
    class Meta:
        model = Transaction
        fields = [
            'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
            'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
            'payment_format', 'is_laundering'
        ]
        widgets = {
            'timestamp': forms.DateTimeInput(attrs={
                'type': 'datetime-local',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'from_bank': forms.NumberInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'from_account': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black; font-family: monospace;'
            }),
            'to_bank': forms.NumberInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'to_account': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black; font-family: monospace;'
            }),
            'amount_received': forms.NumberInput(attrs={
                'step': '0.01',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'receiving_currency': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
                'placeholder': 'e.g., US Dollar, Euro'
            }),
            'amount_paid': forms.NumberInput(attrs={
                'step': '0.01',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'payment_currency': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
                'placeholder': 'e.g., US Dollar, Euro'
            }),
            'payment_format': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
                'placeholder': 'e.g., Wire Transfer, Credit Card'
            }),
            'is_laundering': forms.CheckboxInput(attrs={
                'style': 'width: 16px; height: 16px; border-radius: 4px; border: 1px solid #999;'
            })
        }
        labels = {
            'from_bank': 'Source Bank ID',
            'from_account': 'Source Account',
            'to_bank': 'Destination Bank ID',
            'to_account': 'Destination Account',
            'amount_received': 'Amount Received',
            'amount_paid': 'Amount Paid',
            'is_laundering': 'Flag as Money Laundering'
        }
        help_texts = {
            'timestamp': 'Transaction date and time',
            'from_bank': 'Numeric identifier for the source bank',
            'from_account': 'Account identifier (hexadecimal format)',
            'to_bank': 'Numeric identifier for the destination bank',
            'to_account': 'Account identifier (hexadecimal format)',
            'is_laundering': 'Check if this transaction is known to be money laundering'
        }


class TransactionUpdateForm(forms.ModelForm):
    class Meta:
        model = Transaction
        fields = [
            'timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
            'amount_received', 'receiving_currency', 'amount_paid', 'payment_currency',
            'payment_format', 'is_laundering'
        ]
        widgets = {
            'timestamp': forms.DateTimeInput(attrs={
                'type': 'datetime-local',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'from_bank': forms.NumberInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'from_account': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black; font-family: monospace;'
            }),
            'to_bank': forms.NumberInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'to_account': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black; font-family: monospace;'
            }),
            'amount_received': forms.NumberInput(attrs={
                'step': '0.01',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'receiving_currency': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'amount_paid': forms.NumberInput(attrs={
                'step': '0.01',
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'payment_currency': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'payment_format': forms.TextInput(attrs={
                'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
            }),
            'is_laundering': forms.CheckboxInput(attrs={
                'style': 'width: 16px; height: 16px; border-radius: 4px; border: 1px solid #999;'
            })
        }
        labels = {
            'from_bank': 'Source Bank ID',
            'from_account': 'Source Account',
            'to_bank': 'Destination Bank ID',
            'to_account': 'Destination Account',
            'amount_received': 'Amount Received',
            'amount_paid': 'Amount Paid',
            'is_laundering': 'Flag as Money Laundering'
        }



class ModelTrainingForm(forms.Form):

    MODEL_CHOICES = [
        ('gin', 'Graph Isomorphism Network (GIN)'),
        ('gat', 'Graph Attention Network (GAT)'),
        ('pna', 'Principal Neighborhood Aggregation (PNA)'),
        ('rgcn', 'Relational Graph Convolutional Network (RGCN)'),
        ('autoencoder', 'Graph Autoencoder (GAE)'),
        ('multi-pna-ae', 'Multi-PNA Autoencoder')
    ]
    
    model_type = forms.ChoiceField(
        choices=MODEL_CHOICES,
        label='Model Architecture',
        widget=forms.Select(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
        })
    )

    
    use_edge_mlps = forms.BooleanField(
        initial=True,
        required=False,
        label='Use Edge Updates (MLPs)',
        help_text='Updates edge features using MLPs during message passing',
        widget=forms.CheckboxInput(attrs={
            'style': 'width: 16px; height: 16px; border-radius: 4px; border: 1px solid #999;'
        })
    )
    
    use_reverse_mp = forms.BooleanField(
        initial=True,
        required=False,
        label='Use Reverse Message Passing',
        help_text='Enables bidirectional message passing',
        widget=forms.CheckboxInput(attrs={
            'style': 'width: 16px; height: 16px; border-radius: 4px; border: 1px solid #999;'
        })
    )
    
    use_ports = forms.BooleanField(
        initial=True,
        required=False,
        label='Use Port Numberings',
        help_text='Adds port numbering features to edges',
        widget=forms.CheckboxInput(attrs={
            'style': 'width: 16px; height: 16px; border-radius: 4px; border: 1px solid #999;'
        })
    )
    
    use_ego_ids = forms.BooleanField(
        initial=True,
        required=False,
        label='Use Ego IDs',
        help_text='Adds ego ID features to center nodes in each batch',
        widget=forms.CheckboxInput(attrs={
            'style': 'width: 16px; height: 16px; border-radius: 4px; border: 1px solid #999;'
        })
    )
    
    batch_size = forms.IntegerField(
        initial=4096,
        min_value=128,
        max_value=16384,
        label='Batch Size',
        help_text='Number of edges in each training batch',
        widget=forms.NumberInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
        })
    )
    
    epochs = forms.IntegerField(
        initial=50,
        min_value=1,
        max_value=200,
        label='Training Epochs',
        help_text='Number of complete passes through the training dataset',
        widget=forms.NumberInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
        })
    )
    
    num_neighbors = forms.CharField(
        initial="100 100",
        label='Neighbor Sampling',
        help_text='Space-separated list of neighbors to sample at each hop (descending)',
        widget=forms.TextInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;'
        })
    )
    
    model_name = forms.CharField(
        label='Model Name',
        max_length=100,
        help_text='Name of this model',
        widget=forms.TextInput(attrs={
            'style': 'width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; background-color: white; color: black;',
            'placeholder': 'e.g., gin_fraud_detector_v1'
        })
    )
    
    def clean_num_neighbors(self):
        data = self.cleaned_data['num_neighbors']
        try:
            neighbor_list = [int(x) for x in data.split()]
            if not neighbor_list:
                raise forms.ValidationError("Format is 2 separated integers, by space")
            return neighbor_list
        except ValueError:
            raise forms.ValidationError("Format is 2 separated integers, by space")