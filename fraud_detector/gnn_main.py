import os
import sys
import json
import logging
import traceback
import time
import torch
import numpy as np
import random
from django.conf import settings

sys.path.insert(0, os.path.join(settings.BASE_DIR))
from fraud_detector.model_config_util import create_data_config

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_training(args_dict):
    start_time = time.time()
    
    try:
        args = Args(**args_dict)
        
        logging.info(f"Starting training for model: {args.unique_name}")
        logging.info(f"Model type: {args.model}")
        logging.info(f"Training parameters: {json.dumps(args_dict, default=str)}")
        

        current_dir = os.path.dirname(os.path.abspath(__file__))
        logging.info(f"Current directory: {current_dir}")
        
        gnn_models_dir = os.path.join(current_dir, "gnn_models")
        if os.path.exists(gnn_models_dir):
            logging.info(f"GNN models path: {gnn_models_dir}")
            logging.info(f"GNN models directory found with files: {os.listdir(gnn_models_dir)}")
        else:
            logging.info(f"GNN models directory not found at {gnn_models_dir}")
        
        sys.path.insert(0, gnn_models_dir)
        
        data_config = create_data_config(args.unique_name)
        

        data_config['paths']['aml_data'] = os.path.join(settings.MEDIA_ROOT, 'inference')
        

        try:
            from data_loading import get_data
            from training import train_gnn
            logging.info("Successfully imported data_loading and training modules")
        except ImportError as ie:
            logging.error(f"Error importing modules: {str(ie)}")
            return {
                'success': False,
                'error': f"Error importing modules: {str(ie)}",
                'traceback': traceback.format_exc()
            }
        
        logging.info("Loading transactions")
        try:

            transaction_file_path = os.path.join(data_config['paths']['aml_data'], 'formatted_transactions.csv')
            logging.info(f"Looking for transaction file at: {transaction_file_path}")
            
            if not os.path.exists(transaction_file_path):
                raise FileNotFoundError(f"Transaction file not found at: {transaction_file_path}")
            
            org_data = args.data
            args.data = "" 
            
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
            
            args.data = org_data
            
            logging.info(f"Data loaded successfully. Training samples: {len(tr_inds)}, Validation samples: {len(val_inds)}, Test samples: {len(te_inds)}")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
        set_seed(args.seed)
        
        logging.info("Training started")
        
        model, metrics = train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
        
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")
        
        logging.info(f"Final training metrics:")
        logging.info(f"  Train F1: {metrics['train_f1']:.4f}")
        logging.info(f"  Validation F1: {metrics['val_f1']:.4f}")
        logging.info(f"  Test F1: {metrics['test_f1']:.4f}")
        logging.info(f"  Best Validation F1: {metrics['best_val_f1']:.4f}")
        logging.info(f"  Best Test F1 (corresponding to best val): {metrics['best_test_f1']:.4f}")
        
        checkpoint_path = os.path.join(data_config['paths']['model_to_save'], f'checkpoint_{args.unique_name}.tar')
        if os.path.exists(checkpoint_path):
            logging.info(f"Model saved successfully at: {checkpoint_path}")
            
            return {
                'success': True,
                'training_time': training_time,
                'metrics': metrics
            }
        else:
            logging.warning(f"Model checkpoint not found at: {checkpoint_path}")
            return {
                'success': True,
                'training_time': training_time,
                'metrics': metrics
            }
            
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        training_time = time.time() - start_time if 'start_time' in locals() else 0
        logging.info(f"Training process completed in {training_time:.2f} seconds")

def setup_args_from_form(form_data):
    args = {
        'model': form_data['model_type'],
        'batch_size': form_data['batch_size'],
        'n_epochs': form_data['epochs'],
        'num_neighs': form_data['num_neighbors'],
        'emlps': form_data['use_edge_mlps'],
        'reverse_mp': form_data['use_reverse_mp'],
        'ports': form_data['use_ports'],
        'ego': form_data['use_ego_ids'],
        'tds': False,
        'unique_name': form_data['model_name'],
        'data': 'uploaded_data',
        'seed': 42,
        'save_model': True,
        'testing': True,
        'tqdm': False,
        'inference': False,
        'finetune': False,
        'avg_tps': False
    }
    
    return args