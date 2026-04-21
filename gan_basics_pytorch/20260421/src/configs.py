# import os
import yaml
# import torch


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# def get_device(config):
#     use_cuda = config['train'].get('device', 'cuda')
#     return torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')


def merge_configs(default, override):
    merged = default.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


# def create_dirs(config):
#     dirs = [
#         config['paths']['output_dir'],
#         config['paths']['checkpoint_dir']
#     ]
#     for d in dirs:
#         os.makedirs(d, exist_ok=True)
