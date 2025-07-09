import datetime
import os
from django.conf import settings
from django.core.cache import cache
from django.views.decorators.http import require_POST
from django.http import JsonResponse
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Sum, Q
from django.core.paginator import Paginator
from .models import Transaction
import shap
from .forms import CSVUploadForm, TransactionFilterForm,   TransactionCreateForm, TransactionUpdateForm
import logging
from .csv_import import process_and_import_csv
from django.db import connection
from django.shortcuts import get_object_or_404
from .models import Transaction
import numpy as np
from .forms import ModelTrainingForm
import torch
import sys
from .model_config_util import (
    save_model_config, 
    get_model_config, 
    get_trained_models
)
from .training_manager import start_training_process, get_process_status
from .gnn_main import setup_args_from_form
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pickle
import json

from users.decorators import auditor_required

logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'home.html')

@login_required
@auditor_required
def transactions(request):
    form = TransactionFilterForm(request.GET or None)
    transactions = Transaction.objects.all()
    
    if form.is_valid():
        if form.cleaned_data.get('start_date'):
            start_date = datetime.datetime.combine(form.cleaned_data['start_date'], datetime.time.min)
            transactions = transactions.filter(timestamp__gte=start_date)
        
        if form.cleaned_data.get('end_date'):
            end_date = datetime.datetime.combine(form.cleaned_data['end_date'], datetime.time.max)
            transactions = transactions.filter(timestamp__lte=end_date)
            
        if form.cleaned_data.get('receiving_currency'):
            transactions = transactions.filter(
                receiving_currency__icontains=form.cleaned_data['receiving_currency']
            )
            
        if form.cleaned_data.get('payment_currency'):
            transactions = transactions.filter(
                payment_currency__icontains=form.cleaned_data['payment_currency']
            )
            
        if form.cleaned_data.get('payment_format'):
            transactions = transactions.filter(
                payment_format__icontains=form.cleaned_data['payment_format']
            )
            
        if form.cleaned_data.get('min_amount'):
            transactions = transactions.filter(
                Q(amount_received__gte=form.cleaned_data['min_amount']) | 
                Q(amount_paid__gte=form.cleaned_data['min_amount'])
            )
            
        if form.cleaned_data.get('max_amount'):
            transactions = transactions.filter(
                Q(amount_received__lte=form.cleaned_data['max_amount']) | 
                Q(amount_paid__lte=form.cleaned_data['max_amount'])
            )
            
        if form.cleaned_data.get('fraud_only'):
            transactions = transactions.filter(is_laundering=True)
    
    paginator = Paginator(transactions, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'form': form,
        'page_obj': page_obj,
    }
    
    return render(request, 'fraud_detector/transactions.html', context)

@login_required
@auditor_required
def transaction_upload(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            
            if csv_file.size > 500 * 1024 * 1024:
                messages.error(request, "File too large. Maximum size is 500MB.")
                return redirect('transaction_upload')
            
            try:
                has_header = form.cleaned_data['has_header']
                delimiter = form.cleaned_data['delimiter']
                encoding = form.cleaned_data['encoding']
                
                success_count, error_count, error_messages = process_and_import_csv(
                    csv_file, delimiter, encoding, has_header
                )
                
                if success_count > 0:
                    messages.success(request, f"Successfully imported {success_count} transactions.")
                    if error_count > 0:
                        messages.warning(request, f"Failed to import {error_count} transactions due to errors.")
                        
                        display_count = min(5, len(error_messages))
                        for i in range(display_count):
                            messages.warning(request, error_messages[i])
                        
                        if len(error_messages) > display_count:
                            messages.warning(request, f"... and {len(error_messages) - display_count} more errors. Check the log for details.")
                    
                    return redirect('transactions')
                else:
                    if error_count > 0:
                        messages.error(request, f"Failed to import any transactions. {error_count} errors occurred.")
                        display_count = min(10, len(error_messages))
                        for i in range(display_count):
                            messages.error(request, error_messages[i])
                            
                        if len(error_messages) > display_count:
                            messages.error(request, f"... and {len(error_messages) - display_count} more errors. Check the log for details.")
                    else:
                        messages.error(request, "No transactions were found in the file or all transactions failed to import.")
                    
                    return redirect('transaction_upload')
                
            except Exception as e:
                import traceback
                logger.error(f"Error processing CSV file: {str(e)}")
                logger.error(traceback.format_exc())
                messages.error(request, f"Error processing CSV file: {str(e)}")
                return redirect('transaction_upload')
    else:
        form = CSVUploadForm()
    
    return render(request, 'fraud_detector/transaction_upload.html', {'form': form})

@login_required
@auditor_required
def transaction_detail(request, transaction_id):
    try:
        transaction = Transaction.objects.get(id=transaction_id)
    except Transaction.DoesNotExist:
        messages.error(request, "Transaction not found.")
        return redirect('transactions')
    
    context = {
        'transaction': transaction,
    }
    
    return render(request, 'fraud_detector/transaction_detail.html', context)

@login_required
@auditor_required
def transaction_graph_data(request, transaction_id):
    try:
        depth = min(int(request.GET.get('depth', 1)), 6)
        fraud_only = request.GET.get('fraud_only', 'false').lower() == 'true'
        
        root_transaction = Transaction.objects.get(id=transaction_id)
        
        nodes_dict = {}
        edges_dict = {}
        accounts_to_process = set()
        processed_transactions_ids = set([root_transaction.id])
        fraudulent_accounts = set()
        
        from_account_id = f"account_{root_transaction.from_account}"
        to_account_id = f"account_{root_transaction.to_account}"
        
        known_fraud = root_transaction.is_laundering
        
        nodes_dict[from_account_id] = {
            'id': from_account_id,
            'label': f"{root_transaction.from_account[:8]}...",
            'bank': root_transaction.from_bank,
            'type': 'account',
            'fraud_connection': known_fraud
        }
        
        nodes_dict[to_account_id] = {
            'id': to_account_id,
            'label': f"{root_transaction.to_account[:8]}...",
            'bank': root_transaction.to_bank,
            'type': 'account',
            'fraud_connection': known_fraud
        }
        
        edge_id = f"tx_{root_transaction.id}"
        edges_dict[edge_id] = {
            'id': edge_id,
            'source': from_account_id,
            'target': to_account_id,
            'amount': float(root_transaction.amount_received),
            'currency': root_transaction.receiving_currency,
            'timestamp': root_transaction.timestamp.isoformat(),
            'is_fraud': bool(root_transaction.is_laundering),
            'transaction_id': root_transaction.id
        }
        
        if known_fraud:
            fraudulent_accounts.add(root_transaction.from_account)
            fraudulent_accounts.add(root_transaction.to_account)
        
        if depth > 0:
            accounts_to_process.add(root_transaction.from_account)
            accounts_to_process.add(root_transaction.to_account)
        
        current_depth = 1
        
        accounts_leading_to_fraud = set()
        if known_fraud:
            accounts_leading_to_fraud.add(root_transaction.from_account)
            accounts_leading_to_fraud.add(root_transaction.to_account)
        
        destination_accounts = {}
        
        while current_depth <= depth and accounts_to_process:
            next_level_accounts = set() 
            
            account_filter = Q()
            for account in accounts_to_process:
                account_filter |= Q(from_account=account) | Q(to_account=account)
            
            already_processed = ~Q(id__in=processed_transactions_ids)
            

            transactions = Transaction.objects.filter(
                account_filter & already_processed
            ).order_by('-timestamp')[:200]
            

            fraud_transactions_curr_lvl = []
            
            for transaction in transactions:
                if transaction.id in processed_transactions_ids:
                    continue
                    
                processed_transactions_ids.add(transaction.id)
                
                from_account_id = f"account_{transaction.from_account}"
                to_account_id = f"account_{transaction.to_account}"
                
                destination_accounts[transaction.id] = (transaction.from_account, transaction.to_account)
                
                is_fraudulent = transaction.is_laundering
                
                if is_fraudulent:
                    fraud_transactions_curr_lvl.append(transaction.id)
                    fraudulent_accounts.add(transaction.from_account)
                    fraudulent_accounts.add(transaction.to_account)
                    accounts_leading_to_fraud.add(transaction.from_account)
                    accounts_leading_to_fraud.add(transaction.to_account)
                
                if not fraud_only or is_fraudulent:
                    if from_account_id not in nodes_dict:
                        nodes_dict[from_account_id] = {
                            'id': from_account_id,
                            'label': f"{transaction.from_account[:8]}...",
                            'bank': transaction.from_bank,
                            'type': 'account',
                            'fraud_connection': is_fraudulent
                        }
                    elif is_fraudulent:
                        nodes_dict[from_account_id]['fraud_connection'] = True
                    
                    if to_account_id not in nodes_dict:
                        nodes_dict[to_account_id] = {
                            'id': to_account_id,
                            'label': f"{transaction.to_account[:8]}...",
                            'bank': transaction.to_bank,
                            'type': 'account',
                            'fraud_connection': is_fraudulent
                        }
                    elif is_fraudulent:
                        nodes_dict[to_account_id]['fraud_connection'] = True
                    
                    edge_id = f"tx_{transaction.id}"
                    edges_dict[edge_id] = {
                        'id': edge_id,
                        'source': from_account_id,
                        'target': to_account_id,
                        'amount': float(transaction.amount_received),
                        'currency': transaction.receiving_currency,
                        'timestamp': transaction.timestamp.isoformat(),
                        'is_fraud': bool(transaction.is_laundering),
                        'transaction_id': transaction.id
                    }
                

                if current_depth < depth:
                    next_level_accounts.add(transaction.from_account)
                    next_level_accounts.add(transaction.to_account)
            

            if fraud_only and fraud_transactions_curr_lvl:
                for edge_id, (from_account, to_account) in destination_accounts.items():
                    if edge_id not in edges_dict and (from_account in accounts_leading_to_fraud or to_account in accounts_leading_to_fraud):
                        transaction = Transaction.objects.get(id=edge_id)
                        
                        from_account_id = f"account_{transaction.from_account}"
                        to_account_id = f"account_{transaction.to_account}"
                        
                        if from_account_id not in nodes_dict:
                            nodes_dict[from_account_id] = {
                                'id': from_account_id,
                                'label': f"{transaction.from_account[:8]}...",
                                'bank': transaction.from_bank,
                                'type': 'account',
                                'fraud_connection': transaction.from_account in fraudulent_accounts
                            }
                        
                        if to_account_id not in nodes_dict:
                            nodes_dict[to_account_id] = {
                                'id': to_account_id,
                                'label': f"{transaction.to_account[:8]}...",
                                'bank': transaction.to_bank,
                                'type': 'account',
                                'fraud_connection': transaction.to_account in fraudulent_accounts
                            }
                        
                        edge_id = f"tx_{transaction.id}"
                        edges_dict[edge_id] = {
                            'id': edge_id,
                            'source': from_account_id,
                            'target': to_account_id,
                            'amount': float(transaction.amount_received),
                            'currency': transaction.receiving_currency,
                            'timestamp': transaction.timestamp.isoformat(),
                            'is_fraud': bool(transaction.is_laundering),
                            'transaction_id': transaction.id
                        }
                        
                        accounts_leading_to_fraud.add(from_account)
                        accounts_leading_to_fraud.add(to_account)
            
            accounts_to_process = next_level_accounts
            current_depth += 1
        
        if fraud_only:
            filtered_nodes = {}
            filtered_edges = {}
            
            accounts_in_fraud_paths = set(fraudulent_accounts)
            
            if not accounts_in_fraud_paths:
                accounts_in_fraud_paths.add(root_transaction.from_account)
                accounts_in_fraud_paths.add(root_transaction.to_account)
            
            old_size = 0
            while len(accounts_in_fraud_paths) > old_size:
                old_size = len(accounts_in_fraud_paths)
                
                for edge_id, (from_account, to_account) in destination_accounts.items():
                    if to_account in accounts_in_fraud_paths and from_account not in accounts_in_fraud_paths:
                        accounts_in_fraud_paths.add(from_account)
            

            for account in accounts_in_fraud_paths:
                node_id = f"account_{account}"
                if node_id in nodes_dict:
                    filtered_nodes[node_id] = nodes_dict[node_id]
                    if account in fraudulent_accounts:
                        filtered_nodes[node_id]['fraud_connection'] = True
            
            for edge_id, edge in edges_dict.items():
                source_id = edge['source']
                target_id = edge['target']
                
                if source_id in filtered_nodes and target_id in filtered_nodes:
                    filtered_edges[edge_id] = edge
            
            nodes_dict = filtered_nodes
            edges_dict = filtered_edges
        
        response_data = {
            'nodes': list(nodes_dict.values()),
            'edges': list(edges_dict.values()),
            'debug': {
                'node_count': len(nodes_dict),
                'edge_count': len(edges_dict),
                'depth': depth,
                'transaction_id': transaction_id,
                'processed_transactions': len(processed_transactions_ids),
                'fraud_only': fraud_only,
                'fraudulent_accounts_count': len(fraudulent_accounts)
            }
        }
        
        return JsonResponse(response_data)
    
    except Transaction.DoesNotExist:
        return JsonResponse({'error': 'Transaction not found'}, status=404)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return JsonResponse({
            'error': str(e),
            'trace': error_trace
        }, status=500)
    
@login_required
def transaction_create(request):
    if request.method == 'POST':
        form = TransactionCreateForm(request.POST)
        if form.is_valid():
            transaction = form.save(commit=False)
            transaction.processed = False
            transaction.save()
            messages.success(request, f'Transaction #{transaction.id} created successfully!')
            return redirect('transaction_detail', transaction_id=transaction.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = TransactionCreateForm()
    
    return render(request, 'fraud_detector/transaction_create.html', {'form': form})


@login_required
def transaction_update(request, transaction_id):
    transaction = get_object_or_404(Transaction, id=transaction_id)
    
    if request.method == 'POST':
        form = TransactionUpdateForm(request.POST, instance=transaction)
        if form.is_valid():
            updated_transaction = form.save(commit=False)
            updated_transaction.processed = False
            updated_transaction.save()
            messages.success(request, f'Transaction #{transaction.id} updated successfully!')
            return redirect('transaction_detail', transaction_id=transaction.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        initial_data = {
            'timestamp': transaction.timestamp.strftime('%Y-%m-%dT%H:%M') if transaction.timestamp else None
        }
        form = TransactionUpdateForm(instance=transaction, initial=initial_data)
    
    context = {
        'form': form,
        'transaction': transaction
    }
    
    return render(request, 'fraud_detector/transaction_update.html', context)


@login_required
def transaction_delete(request, transaction_id):
    transaction = get_object_or_404(Transaction, id=transaction_id)
    
    if request.method == 'POST':
        transaction_id = transaction.id
        transaction.delete()
        messages.success(request, f'Transaction #{transaction_id} deleted successfully!')
        return redirect('transactions')
    
    return render(request, 'fraud_detector/transaction_delete_confirm.html', {'transaction': transaction})


@login_required
@require_POST
def process_for_inference(request):
    try:
        transactions = Transaction.objects.all().order_by('timestamp')
        
        if not transactions.exists():
            return JsonResponse({
                'status': 'error',
                'message': 'No transactions found in the database'
            })
        
        output_dir = os.path.join(settings.MEDIA_ROOT, 'inference')
        os.makedirs(output_dir, exist_ok=True)
        
        outPath = os.path.join(output_dir, 'formatted_transactions.csv')
        
        currency = dict()
        paymentFormat = dict()
        account = dict()
        
        def get_dict_val(name, collection):
            if name in collection:
                val = collection[name]
            else:
                val = len(collection)
                collection[name] = val
            return val
        
        header = "EdgeID,from_id,to_id,Timestamp,Amount Sent,Sent Currency,Amount Received,Received Currency,Payment Format,Is Laundering\n"
        timestamp_start = -1
        
        with open(outPath, 'w') as writer:
            writer.write(header)
            
            for i, transaction in enumerate(transactions):
                datetime_object = transaction.timestamp
                ts = datetime_object.timestamp()
                
                if timestamp_start == -1:
                    startTime = datetime.datetime(datetime_object.year, datetime_object.month, datetime_object.day)
                    timestamp_start = startTime.timestamp() - 10
                rel_ts = int(ts - timestamp_start)
                
                curr1 = get_dict_val(transaction.receiving_currency, currency)
                curr2 = get_dict_val(transaction.payment_currency, currency)
                format = get_dict_val(transaction.payment_format, paymentFormat)
                
                fromAccIdStr = str(transaction.from_bank) + str(transaction.from_account)
                fromId = get_dict_val(fromAccIdStr, account)
                toAccIdStr = str(transaction.to_bank) + str(transaction.to_account)
                toId = get_dict_val(toAccIdStr, account)
                
                amountReceivedOrig = float(transaction.amount_received)
                amountPaidOrig = float(transaction.amount_paid)
                isl = 1 if transaction.is_laundering else 0
                
                line = f"{i},{fromId},{toId},{rel_ts},{amountPaidOrig},{curr2},{amountReceivedOrig},{curr1},{format},{isl}\n"
                writer.write(line)
        
        formatted = pd.read_csv(outPath)
        formatted = formatted.sort_values('Timestamp')
        formatted.to_csv(outPath, index=False)
        
        from django.db import transaction as db_transaction
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with db_transaction.atomic():
                    Transaction.objects.all().update(processed=True)
                break
            except Exception as e:
                if 'deadlock' in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                else:
                    raise
        
        cache.set('last_processed_timestamp', datetime.datetime.now().timestamp())
        cache.set('last_processed_count', transactions.count())
        
        return JsonResponse({
            'status': 'success',
            'message': f'Successfully processed {transactions.count()} transactions',
            'path': outPath
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error processing transactions for inference: {str(e)}")
        logger.error(error_trace)
        
        return JsonResponse({
            'status': 'error',
            'message': f'Error processing transactions: {str(e)}'
        }, status=500)

@login_required
def get_processing_status(request):
    try:
        total_count = Transaction.objects.count()
        processed_count = Transaction.objects.filter(processed=True).count()
        unprocessed_count = total_count - processed_count
        
        if total_count == 0:
            status = 'no_transactions'
        elif processed_count == total_count:
            status = 'fully_processed'
        elif processed_count > 0:
            status = 'partially_processed'
        else:
            status = 'none_processed'
        
        return JsonResponse({
            'status': status,
            'total_count': total_count,
            'processed_count': processed_count,
            'unprocessed_count': unprocessed_count,
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@login_required
@auditor_required
def inference_view(request):
    trained_models = get_trained_models()
    
    available_models = []
    for model in trained_models:
        checkpoint_path = os.path.join(settings.MEDIA_ROOT, 'models', f'checkpoint_{model["name"]}.tar')
        if os.path.exists(checkpoint_path):
            available_models.append({
                'name': model['name'],
                'type': model['config']['model_type'],
                'epochs': model['config']['epochs'],
                'modified': datetime.datetime.fromtimestamp(model['modified']).strftime('%Y-%m-%d %H:%M')
            })
    
    data_path = os.path.join(settings.MEDIA_ROOT, 'inference', 'formatted_transactions.csv')
    file_exists = os.path.exists(data_path)
    
    total_count = Transaction.objects.count()
    processed_count = Transaction.objects.filter(processed=True).count()
    
    if not file_exists:
        data_status = 'no_data'
        status_message = 'No processed data found'
        status_color = 'danger'
    elif processed_count < total_count:
        data_status = 'outdated'
        status_message = 'Data outdated.'
        status_color = 'warning'
    else:
        data_status = 'ready'
        status_message = 'Processed data available and up to date.'
        status_color = 'success'
    
    return render(request, 'fraud_detector/inference.html', {
        'available_models': available_models,
        'data_status': data_status,
        'status_message': status_message,
        'status_color': status_color,
        'file_exists': file_exists,
        'total_count': total_count,
        'processed_count': processed_count
    })

@login_required
@require_POST
def run_inference(request):
    try:
        model_name = request.POST.get('model_name')
        transaction_limit = request.POST.get('transaction_limit')
        
        if not model_name:
            return JsonResponse({
                'status': 'error',
                'message': 'No model selected'
            })
        
        model_config = get_model_config(model_name)
        if not model_config:
            return JsonResponse({
                'status': 'error',
                'message': f'Model configuration not found for {model_name}'
            })
        
        checkpoint_path = os.path.join(settings.MEDIA_ROOT, 'models', f'checkpoint_{model_name}.tar')
        if not os.path.exists(checkpoint_path):
            return JsonResponse({
                'status': 'error',
                'message': f'Model checkpoint not found for {model_name}'
            })
        
        data_path = os.path.join(settings.MEDIA_ROOT, 'inference', 'formatted_transactions.csv')
        if not os.path.exists(data_path):
            return JsonResponse({
                'status': 'error',
                'message': 'Formatted transactions data not found. Please process transactions first.'
            })
        
        limit = None
        if transaction_limit:
            try:
                limit = int(transaction_limit)
                if limit <= 0:
                    limit = None
            except ValueError:
                limit = None
        
        gnn_models_dir = os.path.join(settings.BASE_DIR, 'fraud_detector', 'gnn_models')
        sys.path.insert(0, gnn_models_dir)
        
        from .gnn_models.data_loading import get_data
        from .gnn_models.data_util import GraphData, HeteroData
        from .gnn_models.train_util import AddEgoIds, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero
        from .gnn_models.models import GINe, GATe, PNA, RGCN, GraphAutoEncoder, MultiPNAAutoEncoder
        from .gnn_models.training import get_model
        from torch_geometric.nn import to_hetero
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        import tqdm
        
        args_dict = {
            'model': model_config['model_type'],
            'batch_size': model_config.get('batch_size', 8192),
            'n_epochs': model_config.get('epochs', 100),
            'num_neighs': model_config.get('num_neighbors', [100, 100]),
            'emlps': model_config.get('use_edge_mlps', False),
            'reverse_mp': model_config.get('use_reverse_mp', False),
            'ports': model_config.get('use_ports', False),
            'ego': model_config.get('use_ego_ids', False),
            'tds': False,
            'unique_name': model_name,
            'data': '',
            'seed': 42,
            'save_model': False,
            'testing': True,
            'tqdm': False,
            'inference': True,
            'finetune': False,
            'avg_tps': False
        }
        
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        args = Args(**args_dict)
        
        from .model_config_util import create_data_config
        data_config = create_data_config(model_name)
        data_config['paths']['aml_data'] = os.path.join(settings.MEDIA_ROOT, 'inference')
        
        logging.info("Loading transaction data for inference...")
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
        
        if limit and limit < len(te_inds):
            te_inds = te_inds[:limit]
            logging.info(f"Limiting inference to first {limit} test transactions")
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        transform = AddEgoIds() if args.ego else None
        
        add_arange_ids([tr_data, val_data, te_data])
        
        # Get data loaders
        tr_loader, val_loader, te_loader = get_loaders(
            tr_data, val_data, te_data, 
            tr_inds, val_inds, te_inds, 
            transform, args
        )
        
        sample_batch = next(iter(tr_loader))
        
        class Config:
            def __init__(self, model_config, model_type):
                import json
                model_settings_path = os.path.join(gnn_models_dir, 'model_settings.json')
                
                self.n_gnn_layers = 2
                self.n_hidden = 64
                self.dropout = 0.5
                self.final_dropout = 0.5
                self.lr = 0.0001
                self.w_ce1 = 1.0
                self.w_ce2 = 7.1
                self.n_heads = None
                self.latent_dim = None
                
                try:
                    with open(model_settings_path, 'r') as f:
                        model_settings = json.load(f)
                        
                    if model_type in model_settings:
                        params = model_settings[model_type].get('params', {})
                        
                        self.lr = params.get('lr', self.lr)
                        self.n_hidden = int(params.get('n_hidden', self.n_hidden))
                        self.n_gnn_layers = int(params.get('n_gnn_layers', self.n_gnn_layers))
                        self.dropout = params.get('dropout', self.dropout)
                        self.final_dropout = params.get('final_dropout', self.final_dropout)
                        self.w_ce1 = params.get('w_ce1', self.w_ce1)
                        self.w_ce2 = params.get('w_ce2', self.w_ce2)
                        
                        if model_type == 'gat':
                            self.n_heads = int(params.get('n_heads', 4))
                        
                        if model_type in ['autoencoder', 'multi-pna-ae']:
                            self.latent_dim = int(params.get('latent_dim', 32 if model_type == 'autoencoder' else 12))
                            
                except Exception as e:
                    logging.warning(f"Could not load model settings: {str(e)}")
        
        config = Config(model_config, model_config['model_type'])
        
        model = get_model(sample_batch, config, args)
        
        if args.reverse_mp and model_config['model_type'] not in ['autoencoder', 'multi-pna-ae']:
            model = to_hetero(model, te_data.metadata(), aggr='mean')
        
        logging.info(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logging.info("Running inference on test transactions...")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm.tqdm(te_loader, disable=True):
                if not args.reverse_mp:
                    inds = te_inds.detach().cpu()
                    batch_edge_inds = inds[batch.input_id.detach().cpu()]
                    batch_edge_ids = te_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
                    mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
                    
                    missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())
                    
                    if missing.sum() != 0:
                        missing_ids = batch_edge_ids[missing].int()
                        n_ids = batch.n_id
                        add_edge_index = te_data.edge_index[:, missing_ids].detach().clone()
                        node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
                        add_edge_index = torch.tensor([[node_mapping.get(val.item(), 0) for val in row] for row in add_edge_index])
                        add_edge_attr = te_data.edge_attr[missing_ids, :].detach().clone()
                        add_y = te_data.y[missing_ids].detach().clone()
                    
                        batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
                        batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
                        batch.y = torch.cat((batch.y, add_y), 0)
                        
                        mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))
                    
                    batch.edge_attr = batch.edge_attr[:, 1:]
                    batch.to(device)
                    
                    out = model(batch.x, batch.edge_index, batch.edge_attr)
                    out = out[mask]
                    pred = out.argmax(dim=-1)
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(batch.y[mask].cpu().numpy())
                    
                else:
                    inds = te_inds.detach().cpu()
                    batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
                    batch_edge_ids = te_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
                    mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
                    
                    missing = ~torch.isin(batch_edge_ids, batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu())
                    
                    if missing.sum() != 0:
                        missing_ids = batch_edge_ids[missing].int()
                        n_ids = batch['node'].n_id
                        add_edge_index = te_data['node', 'to', 'node'].edge_index[:, missing_ids].detach().clone()
                        node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
                        add_edge_index = torch.tensor([[node_mapping.get(val.item(), 0) for val in row] for row in add_edge_index])
                        add_edge_attr = te_data['node', 'to', 'node'].edge_attr[missing_ids, :].detach().clone()
                        add_y = te_data['node', 'to', 'node'].y[missing_ids].detach().clone()
                    
                        batch['node', 'to', 'node'].edge_index = torch.cat((batch['node', 'to', 'node'].edge_index, add_edge_index), 1)
                        batch['node', 'to', 'node'].edge_attr = torch.cat((batch['node', 'to', 'node'].edge_attr, add_edge_attr), 0)
                        batch['node', 'to', 'node'].y = torch.cat((batch['node', 'to', 'node'].y, add_y), 0)
                        
                        mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))
                    
                    batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
                    batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
                    batch.to(device)
                    
                    out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
                    out = out[('node', 'to', 'node')]
                    out = out[mask]
                    pred = out.argmax(dim=-1)
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(batch['node', 'to', 'node'].y[mask].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        flagged_count = int(np.sum(all_preds))
        processed_count = len(all_preds)
        true_fraud_count = int(np.sum(all_labels))
        
        logging.info(f"Inference completed. Processed {processed_count} transactions, flagged {flagged_count} as fraudulent")
        logging.info(f"Ground truth: {true_fraud_count} fraudulent transactions")
        logging.info(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return JsonResponse({
            'status': 'success',
            'message': f'Inference completed on {processed_count} test transactions',
            'results': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'processed_transactions': processed_count,
                'flagged_transactions': flagged_count,
                'actual_fraudulent': true_fraud_count,
                'transaction_limit': limit if limit else 'All test transactions'
            }
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error running inference: {str(e)}")
        logger.error(error_trace)
        
        return JsonResponse({
            'status': 'error',
            'message': f'Error running inference: {str(e)}'
        }, status=500)


def about_view(request):
    return render(request, 'about.html')

@login_required
@auditor_required
def training_view(request):
    trained_models = get_trained_models()
    
    csv_path = os.path.join(settings.MEDIA_ROOT, 'inference', 'formatted_transactions.csv')
    file_exists = os.path.exists(csv_path)
    
    total_count = Transaction.objects.count()
    processed_count = Transaction.objects.filter(processed=True).count()
    
    data_ready = file_exists and (processed_count == total_count or total_count == 0)
    
    if not file_exists:
        data_status_message = "No processed data file found."
    elif processed_count < total_count:
        data_status_message = " Process all transactions before training."
    else:
        data_status_message = "Data ready for training."
    
    if request.method == 'POST':
        form = ModelTrainingForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            
            existing_config = get_model_config(form_data['model_name'])
            if existing_config:
                messages.error(request, f"A model with the name '{form_data['model_name']}' already exists. Please choose a different name.")
                return render(request, 'fraud_detector/training.html', {
                    'form': form, 
                    'trained_models': trained_models,
                    'data_ready': data_ready,
                    'data_status_message': data_status_message,
                    'file_exists': file_exists,
                    'processed_count': processed_count,
                    'total_count': total_count
                })
            
            if not data_ready:
                messages.error(request, "Transaction data is not ready.")
                return render(request, 'fraud_detector/training.html', {
                    'form': form, 
                    'trained_models': trained_models,
                    'data_ready': data_ready,
                    'data_status_message': data_status_message,
                    'file_exists': file_exists,
                    'processed_count': processed_count,
                    'total_count': total_count
                })
            
            args_dict = setup_args_from_form(form_data)
            
            save_model_config(form_data['model_name'], {
                'model_type': form_data['model_type'],
                'batch_size': form_data['batch_size'],
                'epochs': form_data['epochs'],
                'num_neighbors': form_data['num_neighbors'],
                'use_edge_mlps': form_data['use_edge_mlps'],
                'use_reverse_mp': form_data['use_reverse_mp'],
                'use_ports': form_data['use_ports'],
                'use_ego_ids': form_data['use_ego_ids'],
            })
            
            success, message = start_training_process(form_data['model_name'], args_dict)
            
            if success:
                messages.success(request, "Training process started successfully.")
                return redirect('model_training_status', model_name=form_data['model_name'])
            else:
                messages.error(request, f"Failed to start training process: {message}")
    else:
        form = ModelTrainingForm()
    
    return render(request, 'fraud_detector/training.html', {
        'form': form, 
        'trained_models': trained_models,
        'data_ready': data_ready,
        'data_status_message': data_status_message,
        'file_exists': file_exists,
        'processed_count': processed_count,
        'total_count': total_count
    })

@login_required
def model_training_status(request, model_name):
    model_config = get_model_config(model_name)
    
    if not model_config:
        messages.error(request, f"Model configuration not found for '{model_name}'")
        return redirect('training')
    
    status = get_process_status(model_name)
    
    status_json = json.dumps(status) if status else 'null'
    
    return render(request, 'fraud_detector/training_status.html', {
        'model_name': model_name,
        'model_config': model_config,
        'status': status,
        'status_json': status_json
    })

@login_required
def model_training_status_api(request, model_name):
    status = get_process_status(model_name)
    
    if status:
        return JsonResponse(status)
    else:
        return JsonResponse({
            'model_name': model_name,
            'status': 'not_found',
            'error': 'Training process not found',
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'log_messages': [],
            'train_f1': 0,
            'val_f1': 0,
            'test_f1': 0
        }, status=404)

@login_required
def process_data_status(request):
    csv_path = os.path.join(settings.MEDIA_ROOT, 'inference', 'formatted_transactions.csv')
    file_exists = os.path.exists(csv_path)
    
    total_count = Transaction.objects.count()
    processed_count = Transaction.objects.filter(processed=True).count()
    
    data_ready = file_exists and (processed_count == total_count or total_count == 0)
    
    return JsonResponse({
        'data_ready': data_ready,
        'file_exists': file_exists,
        'total_count': total_count,
        'processed_count': processed_count,
        'unprocessed_count': total_count - processed_count
    })
def train_xgboost_model(file_path):
    try:
        import xgboost as xgb
    except ImportError:
        logging.error("XGBoost is not installed'")
        raise
    
    df = pd.read_csv(file_path)
    
    target_column = 'Is Laundering'
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    logging.info(f"Train set fraud distribution: {y_train.mean():.4f}")
    logging.info(f"Validation set fraud distribution: {y_val.mean():.4f}")
    logging.info(f"Test set fraud distribution: {y_test.mean():.4f}")
    params = {
        'objective': 'binary:logistic',
        'verbosity': 0,
        'seed': 42,
        'max_depth': 7,
        'learning_rate': 0.027767106691963472,
        'lambda': 0.08108301668277595,
        'scale_pos_weight': 6.710705994948692,
        'colsample_bytree': 0.7405695219739028,
        'subsample': 0.8874337790131545
    }

    model = xgb.XGBClassifier(**params, n_estimators=881)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    
    minority_f1 = f1_score(y_test, y_pred, pos_label=1)
    
    metrics = {
        'minority_f1': float(minority_f1),
        'fraud_ratio': float(y.mean())
    }
    
    return model, X_train, y_train, X_val, y_val, X_test, y_test, metrics, X.columns.tolist()

def compute_shap_values(model, X, X_sample, explainer_type='TreeExplainer', feature_names=None, edge_id=None, max_samples=None):

    X_sample_np = X_sample.values
    
    if max_samples is not None and X_sample.shape[0] > max_samples:
        X_sample = X_sample.iloc[:max_samples]
        X_sample_np = X_sample_np[:max_samples]
        logging.info(f"Limiting SHAP analysis to first {max_samples} samples")
    else:
        logging.info(f"Using all {X_sample.shape[0]} samples for SHAP analysis")
    
    summary_plot_b64 = None
    dependence_plot_b64 = None
    waterfall_plot_b64 = None
    explained_transaction = None
    
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    
    X_sample_np = X_sample.values

    if max_samples is not None and X_sample.shape[0] > max_samples:
        X_sample = X_sample.iloc[:max_samples]
        X_sample_np = X_sample_np[:max_samples]
        logging.info(f"Limiting SHAP analysis to first {max_samples} samples")
    else:
        logging.info(f"Using all {X_sample.shape[0]} samples for SHAP analysis")
    
    plt.style.use('dark_background')
    
    target_idx = None
    if edge_id is not None:
        found = False

        if 'EdgeID' in X_sample.columns:
            matches = X_sample[X_sample['EdgeID'] == edge_id]
            if not matches.empty:
                target_idx = X_sample.index.get_loc(matches.index[0])
                found = True
        elif edge_id in X_sample.index:
            target_idx = X_sample.index.get_loc(edge_id)
            found = True
        
        if found:
            logging.info(f"Found transaction with edge_id {edge_id} at position {target_idx} in the sample")
    
    predictions = model.predict(X_sample_np)
    
    if target_idx is None:
        fraud_indices = np.where(predictions == 1)[0]
        if len(fraud_indices) > 0:
            target_idx = fraud_indices[0]
            logging.info(f"Using first fraudulent transaction at position {target_idx}")
        else:
            target_idx = 0
            logging.info(f"No fraudulent transactions found. Using first transaction.")
            
    if explainer_type == 'TreeExplainer':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_np)
        expected_value = explainer.expected_value
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', show=False)
        summary_plot_b64 = fig_to_base64(plt.gcf())
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='dot', show=False)
        dependence_plot_b64 = fig_to_base64(plt.gcf())
        
        plt.figure(figsize=(10, 6))
        try:
            idx = target_idx
            is_fraud = bool(predictions[idx])
            

            if 'EdgeID' in X_sample.columns:
                transaction_id = X_sample.iloc[idx]['EdgeID']
            else:
                transaction_id = idx
                
            explained_transaction = {
                'edge_id': transaction_id,
                'is_fraud': is_fraud
            }
            
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value, 
                shap_values[idx], 
                feature_names=feature_names, 
                show=False,
                max_display=10
            )
            
            status = "Fraudulent" if is_fraud else "Non-Fraudulent"
            plt.title(f"Explanation for {status} Transaction (ID: {transaction_id})")
        except Exception as e:
            logging.error(f"Error creating waterfall plot: {str(e)}")
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value, 
                shap_values[0], 
                feature_names=feature_names, 
                show=False, 
                max_display=10
            )
            plt.title(f"Transaction explanation")
            

            if 'EdgeID' in X_sample.columns:
                transaction_id = X_sample.iloc[0]['EdgeID']
            else:
                transaction_id = 0
                
            explained_transaction = {
                'edge_id': transaction_id,
                'is_fraud': bool(predictions[0]) if len(predictions) > 0 else False
            }
        
        waterfall_plot_b64 = fig_to_base64(plt.gcf())
    
    plt.style.use('default')
    
    shap_plots = {
        'summary_plot': summary_plot_b64,
        'dependence_plot': dependence_plot_b64,
        'waterfall_plot': waterfall_plot_b64
    }
    
    return shap_plots, shap_values, expected_value, explained_transaction

@login_required
def explainability_view(request):
    gfp_file_path = os.path.join(settings.MEDIA_ROOT, 'inference', 'GFP_transactions.csv')
    exists = os.path.exists(gfp_file_path)
    
    model_dir = os.path.join(settings.MEDIA_ROOT, 'models', 'xgboost')
    model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    model_exists = os.path.exists(model_path)
    
    metadata_path = os.path.join(model_dir, 'xgboost_metadata.json')
    
    model_trained = model_exists
    training_result = None
    feature_importance = None
    shap_plots = None
    explained_transaction = None
    explainer_types = ['TreeExplainer']
    selected_explainer = request.POST.get('explainer', 'TreeExplainer')
    
    specific_edge_id = request.POST.get('edge_id') or request.GET.get('edge_id')
    if specific_edge_id:
        try:
            specific_edge_id = int(specific_edge_id)
        except ValueError:
            specific_edge_id = None
            messages.warning(request, "Invalid edge ID format. Using first fraudulent transaction instead.")
    
    max_samples = request.POST.get('max_samples') or request.GET.get('max_samples')
    if max_samples:
        try:
            max_samples = int(max_samples)
            if max_samples <= 0:
                max_samples = None
                messages.warning(request, "Invalid max samples value. Using all available samples.")
        except ValueError:
            max_samples = None
            messages.warning(request, "Invalid max samples format. Using all available samples.")
    
    if model_exists and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                feature_importance = metadata.get('feature_importance')
                
                training_result = {
                    'metrics': metadata.get('metrics', {}),
                    'model_path': model_path,
                    'feature_count': len(metadata.get('feature_names', [])),
                    'training_samples': metadata.get('training_samples', 0),
                    'validation_samples': metadata.get('validation_samples', 0),
                    'test_samples': metadata.get('test_samples', 0)
                }
        except Exception as e:
            logging.error(f"Error loading model metadata: {str(e)}", exc_info=True)
    
    if request.method == 'POST' and 'train_model' in request.POST and exists:
        try:
            model, X_train, y_train, X_val, y_val, X_test, y_test, metrics, feature_names = train_xgboost_model(gfp_file_path)
            
            os.makedirs(model_dir, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            feature_importance = {name: float(score) for name, score in zip(feature_names, model.feature_importances_)}
            
            metadata = {
                'feature_names': feature_names,
                'feature_importance': feature_importance,
                'metrics': metrics,
                'training_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0],
                'test_samples': X_test.shape[0],
                'trained_at': datetime.datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            training_result = {
                'metrics': metrics,
                'model_path': model_path,
                'feature_count': len(feature_names),
                'training_samples': X_train.shape[0],
                'validation_samples': X_val.shape[0],
                'test_samples': X_test.shape[0]
            }
            
            model_trained = True
            
            messages.success(request, f"XGBoost model trained successfully. F1: {metrics['minority_f1']:.4f}")
            
        except Exception as e:
            messages.error(request, f"Error training model: {str(e)}")
            logging.error(f"Error training XGBoost model: {str(e)}", exc_info=True)
    
    elif request.method == 'POST' and 'generate_shap' in request.POST and model_exists:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                feature_names = metadata.get('feature_names', [])
            
            df = pd.read_csv(gfp_file_path)
            target_column = 'Is Laundering'
            if target_column not in df.columns:
                for col in df.columns:
                    if 'launder' in col.lower() or 'fraud' in col.lower():
                        target_column = col
                        break
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in ['EdgeID']:
                    X[col] = pd.Categorical(X[col]).codes
            
            sample_size = min(10000, X.shape[0])
            
            if specific_edge_id is not None and 'EdgeID' in X.columns:
                specific_mask = X['EdgeID'] == specific_edge_id
                if specific_mask.any():
                    specific_idx = X[specific_mask].index[0]
                    
                    if max_samples and max_samples < sample_size:
                        other_indices = list(X.index[X.index != specific_idx])
                        sample_indices = [specific_idx] + list(np.random.choice(other_indices, 
                                                                            min(sample_size-1, len(other_indices)), 
                                                                            replace=False))
                        X_sample = X.iloc[sample_indices].copy()
                    else:
                        other_indices = list(X.index[X.index != specific_idx])
                        sample_indices = [specific_idx] + list(np.random.choice(other_indices, 
                                                                            min(sample_size-1, len(other_indices)), 
                                                                            replace=False))
                        X_sample = X.iloc[sample_indices].copy()
                    
                    logging.info(f"Created sample with specific transaction {specific_edge_id} at index {specific_idx}")
                    logging.info(f"Sample size: {len(X_sample)}, includes EdgeID: {specific_edge_id in X_sample['EdgeID'].values}")
                else:
                    messages.warning(request, f"Transaction with ID {specific_edge_id} not found in the data.")
                    X_sample = X.sample(sample_size, random_state=42)
            else:
                X_sample = X.sample(sample_size, random_state=42)
            
            shap_plots, _, _, explained_transaction = compute_shap_values(
                model, X, X_sample, selected_explainer, feature_names, specific_edge_id, max_samples
            )

            
            samples_msg = ""
            if max_samples:
                samples_msg = f" using {max_samples} samples"
            transaction_msg = f" for transaction ID {explained_transaction['edge_id']}" if explained_transaction else ""
            
            messages.success(request, f"SHAP explanation generated successfully{transaction_msg}{samples_msg} using {selected_explainer}")
            
        except Exception as e:
            messages.error(request, f"Error generating SHAP values: {str(e)}")
            logging.error(f"Error generating SHAP values: {str(e)}", exc_info=True)
    
    context = {
        'file_exists': exists,
        'model_exists': model_exists,
        'model_trained': model_trained,
        'training_result': training_result,
        'feature_importance': json.dumps(feature_importance) if feature_importance else None,
        'shap_plots': shap_plots,
        'explained_transaction': explained_transaction,
        'explainer_types': explainer_types,
        'selected_explainer': selected_explainer,
        'specific_edge_id': specific_edge_id,
        'max_samples': max_samples
    }
    
    return render(request, 'fraud_detector/explainability.html', context)