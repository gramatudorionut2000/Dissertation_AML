import os
import json
from django.conf import settings

MODELS_DIR = os.path.join(settings.MEDIA_ROOT, 'models')
CONFIG_PATH = os.path.join(MODELS_DIR, 'model_configs.json')

def create_model_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as f:
            json.dump({}, f)

def save_model_config(model_name, config_dict):
    create_model_directories()
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            configs = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        configs = {}
    
    configs[model_name] = config_dict
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(configs, f, indent=2)
    
    return True

def get_model_config(model_name):

    create_model_directories()
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            configs = json.load(f)
        return configs.get(model_name, None)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def get_models():
    create_model_directories()
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            configs = json.load(f)
        return configs
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def get_trained_models():
    create_model_directories()
    
    configs = get_models()
    trained_models = []
    
    for model_name in configs:
        checkpoint_path = os.path.join(MODELS_DIR, f'checkpoint_{model_name}.tar')
        if os.path.exists(checkpoint_path):
            trained_models.append({
                'name': model_name,
                'config': configs[model_name],
                'size': os.path.getsize(checkpoint_path),
                'modified': os.path.getmtime(checkpoint_path)
            })
    
    return trained_models

def create_data_config(model_name):
    
    create_model_directories()
    
    config = {
        "paths": {
            "aml_data": os.path.join(settings.MEDIA_ROOT, 'inference'),
            "model_to_load": MODELS_DIR,
            "model_to_save": MODELS_DIR
        }
    }
    
    return config