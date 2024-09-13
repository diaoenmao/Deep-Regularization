import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training settings
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.1
N = 60000

# Model settings
MODEL_TYPE = 'cnn3'
DATASET = 'cifar10'

# Optimizer settings
OPTIMIZER_TYPE = 'ADMM_Adam_layer'
BETA1 = 0.9
BETA2 = 0.999

# Lottery Ticket settings
LOTTERY_TICKET = {
    'enabled': False,
    'prune_percent': 20,
    'prune_iterations': 10
}

# Paths
SAVE_DIR = 'results'
METRICS_DIR = os.path.join(SAVE_DIR, 'metrics')
PLOTS_DIR = os.path.join(SAVE_DIR, 'plots')

# Create directories if they don't exist
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Additional settings for running multiple experiments
RUN_MULTIPLE_EXPERIMENTS = False
NUM_EXPERIMENTS = 1
C=0.5

# C scheduler settings
C_SCHEDULER = {
    'strategy': 'sine',
    'start_value': 0.01,
    'end_value': 2.01,
    'total_experiments': NUM_EXPERIMENTS
}