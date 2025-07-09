import os
import sys
import subprocess
import json
import logging
import threading
import traceback
from django.conf import settings
from django.utils import timezone
from django.core.cache import cache

sys.path.insert(0, settings.BASE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

ACTIVE_TRAINING_PROCESSES = {}

class TrainingProcess:
    
    def __init__(self, model_name, model_args):
        self.model_name = model_name
        self.model_args = model_args
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.progress = 0
        self.current_epoch = 0
        self.total_epochs = model_args.get('n_epochs', 50)
        self.log_messages = []
        self.error = None
        self.train_f1 = 0
        self.val_f1 = 0
        self.test_f1 = 0
        self.best_val_f1 = 0
        self.best_test_f1 = 0
        self.thread = None
        self.log_handler = None
    
    def start(self):
        self.start_time = timezone.now()
        self.status = "running"
        
        self._add_log("Starting training process for model: " + self.model_name)
        self._add_log(f"Model type: {self.model_args.get('model')}")
        self._add_log(f"Training parameters: {json.dumps({k: v for k, v in self.model_args.items() if k != 'unique_name'})}")
        
        self.thread = threading.Thread(target=self._run_training)
        self.thread.daemon = True
        self.thread.start()
        
        ACTIVE_TRAINING_PROCESSES[self.model_name] = self
        
        self._update_cache()
        
        return True
    
    def _add_log(self, message, level="INFO"):
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"{timestamp} - {level} - {message}"
        self.log_messages.append(full_message)
        
        print(full_message)
        
        self._update_cache()
    
    def _run_training(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self._add_log(f"Current directory: {current_dir}")
            
            gnn_main_path = os.path.join(current_dir, "gnn_main.py")
            if not os.path.exists(gnn_main_path):
                gnn_main_path = os.path.join(settings.BASE_DIR, "fraud_detector", "gnn_main.py")
                
            if not os.path.exists(gnn_main_path):
                self._add_log(f"Cannot find gnn_main.py at: {gnn_main_path}", "ERROR")
                
                self._add_log("Searching for gnn_main.py")
                for root, dirs, files in os.walk(settings.BASE_DIR):
                    if "gnn_main.py" in files:
                        gnn_main_path = os.path.join(root, "gnn_main.py")
                        self._add_log(f"Found gnn_main.py at: {gnn_main_path}")
                        break
                        
            if os.path.exists(gnn_main_path):
                self._add_log(f"Found gnn_main.py at: {gnn_main_path}")
            else:
                raise ImportError(f"Cannot find gnn_main.py")
            
            self._add_log("Starting_training")
            
            try:
                args_file = os.path.join(settings.MEDIA_ROOT, f'training_args_{self.model_name}.json')
                with open(args_file, 'w') as f:
                    json.dump(self.model_args, f)
                
                runner_script = os.path.join(settings.MEDIA_ROOT, f'run_training_{self.model_name}.py')
                with open(runner_script, 'w') as f:
                    f.write(f'''
import os
import sys
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

sys.path.insert(0, r'{settings.BASE_DIR}')

with open(r'{args_file}', 'r') as f:
    args = json.load(f)

logging.info(f"Loaded training arguments: {{args}}")

sys.path.append(r'{os.path.dirname(gnn_main_path)}')
from gnn_main import run_training

logging.info("Starting training...")
result = run_training(args)
logging.info(f"Training completed with result: {{result}}")

with open(r'{os.path.join(settings.MEDIA_ROOT, f"training_result_{self.model_name}.json")}', 'w') as f:
    json.dump(result, f)
''')
                
                self._add_log(f"Executing runner script: {runner_script}")
                
                process = subprocess.Popen(
                    [sys.executable, runner_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        self._add_log(line)
                        
                        if "Epoch" in line:
                            try:
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if part == "Epoch":
                                        epoch_str = parts[i+1]
                                        if '/' in epoch_str:
                                            epoch = int(epoch_str.split('/')[0])
                                        else:
                                            epoch = int(epoch_str)
                                        self.current_epoch = epoch
                                        self.progress = min(100, int(epoch * 100 / self.total_epochs))
                                        break
                            except Exception as e:
                                self._add_log(f"Error parsing epoch info: {str(e)}", "ERROR")
                        
                        if "Train F1:" in line:
                            try:
                                f1_str = line.split("Train F1:")[1].strip()
                                self.train_f1 = float(f1_str)
                            except Exception as e:
                                self._add_log(f"Error parsing train F1: {str(e)}", "ERROR")
                                
                        if "Validation F1:" in line:
                            try:
                                f1_str = line.split("Validation F1:")[1].strip()
                                self.val_f1 = float(f1_str)
                            except Exception as e:
                                self._add_log(f"Error parsing validation F1: {str(e)}", "ERROR")
                                
                        if "Test F1:" in line:
                            try:
                                f1_str = line.split("Test F1:")[1].strip()
                                self.test_f1 = float(f1_str)
                            except Exception as e:
                                self._add_log(f"Error parsing test F1: {str(e)}", "ERROR")
                        
                        if "New best validation F1:" in line:
                            try:
                                parts = line.split("New best validation F1:")[1].split(",")
                                self.best_val_f1 = float(parts[0].strip())
                                if "corresponding test F1:" in parts[1]:
                                    self.best_test_f1 = float(parts[1].split("corresponding test F1:")[1].strip())
                            except Exception as e:
                                self._add_log(f"Error parsing best validation F1: {str(e)}", "ERROR")
                
                process.wait()
                
                result_file = os.path.join(settings.MEDIA_ROOT, f'training_result_{self.model_name}.json')
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                        
                    if result and result.get('success') and 'metrics' in result:
                        metrics = result['metrics']
                        self.train_f1 = metrics.get('train_f1', self.train_f1)
                        self.val_f1 = metrics.get('val_f1', self.val_f1)
                        self.test_f1 = metrics.get('test_f1', self.test_f1)
                        self.best_val_f1 = metrics.get('best_val_f1', self.best_val_f1)
                        self.best_test_f1 = metrics.get('best_test_f1', self.best_test_f1)
                        
                        self._add_log(f"Final metrics loaded from result file:", "INFO")
                        self._add_log(f"  Train F1: {self.train_f1:.4f}", "INFO")
                        self._add_log(f"  Validation F1: {self.val_f1:.4f}", "INFO")
                        self._add_log(f"  Test F1: {self.test_f1:.4f}", "INFO")
                        self._add_log(f"  Best Validation F1: {self.best_val_f1:.4f}", "INFO")
                        self._add_log(f"  Best Test F1: {self.best_test_f1:.4f}", "INFO")
                    else:
                        self._add_log(f"Warning: No metrics found in result file or training failed", "WARNING")
                        if result and not result.get('success'):
                            self._add_log(f"Training error: {result.get('error', 'Unknown error')}", "ERROR")
                else:
                    self._add_log(f"Warning: No result file found at {result_file}", "WARNING")
                
                try:
                    os.remove(args_file)
                    os.remove(runner_script)
                    if os.path.exists(result_file):
                        os.remove(result_file)
                except Exception as e:
                    self._add_log(f"Error cleaning up files: {str(e)}", "WARNING")
                
                if process.returncode == 0:
                    self.status = "completed"
                    self.progress = 100 
                    self._add_log("Training completed successfully")
                else:
                    self.status = "failed"
                    self.error = f"Training process exited with code {process.returncode}"
                    self._add_log(f"Training process exited with code {process.returncode}", "ERROR")
                
            except Exception as e:
                self._add_log(f"Error executing training: {str(e)}", "ERROR")
                self._add_log(traceback.format_exc(), "ERROR")
                raise
        
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self._add_log(f"Training error: {str(e)}", "ERROR")
            self._add_log(traceback.format_exc(), "ERROR")
        
        finally:
            self.end_time = timezone.now()
            self._update_cache()
    
    def _update_cache(self):
        cache_key = f"training_process_{self.model_name}"
        cache.set(cache_key, self.get_status(), timeout=86400)  
    
    def get_status(self):
        duration = None
        if self.start_time:
            end = self.end_time or timezone.now()
            duration = (end - self.start_time).total_seconds()
            
        return {
            'model_name': self.model_name,
            'status': self.status,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': duration,
            'log_messages': self.log_messages[-50:],
            'error': self.error,
            'train_f1': self.train_f1,
            'val_f1': self.val_f1,
            'test_f1': self.test_f1,
            'best_val_f1': self.best_val_f1,
            'best_test_f1': self.best_test_f1
        }

def get_process_status(model_name):
    if model_name in ACTIVE_TRAINING_PROCESSES:
        return ACTIVE_TRAINING_PROCESSES[model_name].get_status()
    
    cache_key = f"training_process_{model_name}"
    status = cache.get(cache_key)
    if status:
        return status
    
    return None

def start_training_process(model_name, args_dict):
    if model_name in ACTIVE_TRAINING_PROCESSES and ACTIVE_TRAINING_PROCESSES[model_name].status == "running":
        return False, "A training process for this model is already running"
    
    process = TrainingProcess(model_name, args_dict)
    success = process.start()
    
    return success, "Training process started successfully"