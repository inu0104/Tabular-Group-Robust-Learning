import argparse
import random
import torch
import json
import sys

import numpy as np

from utils.data_loader import load_data
from method.d3m import run_d3m
from method.group_dro import run_group_dro
from method.gsr import run_gsr
from models import node, tabnet, deepfm, autoint

MODEL_CLASSES = {
    'node': node.NODE,
    'tabnet': tabnet.TabNet,
    'deepfm': deepfm.DeepFM,
    'autoint': autoint.AutoInt
}
METHODS = {
    'd3m': run_d3m,
    'group_dro' : run_group_dro,
    'gsr': run_gsr
}

def main(config_file):
    
    with open(config_file, 'r') as f:
        config = json.load(f)

    seed = config.get("seed", 2025)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, train_df = load_data(config)
   
    model_class = MODEL_CLASSES[config['model_type']]
    model = model_class(config['model_params']).to(device)
    method_fn = METHODS[config["method"]]  
    dataset = config['dataset']
    _ = method_fn(model, train_loader, valid_loader, test_loader, train_df, config['train_params'], device, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments with optional post-processing.')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    main(args.config)
