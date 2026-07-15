# config.py
"""Configuration parameters for LCA system"""

# Video configuration knobs
VIDEO_CONFIGS = {
    'resolutions': [1, 0.64, 0.36, 0.25, 0.16],
    'frame_rates': [1.0, 0.5, 0.3333, 0.2],
    'qp_values': [1, 1.2174, 1.4348, 1.6522, 1.8696]  # Quantization parameters
}

# Total number of configurations (4 * 4 * 5 = 80, but paper uses 125)
# We'll use a subset for practical implementation
TOTAL_CONFIGS = len(VIDEO_CONFIGS['resolutions']) * \
                len(VIDEO_CONFIGS['frame_rates']) * \
                len(VIDEO_CONFIGS['qp_values'])

# Network parameters
BANDWIDTH_MIN = 0.2  # Mbps
BANDWIDTH_MAX = 6.0  # Mbps
CHUNK_DURATION = 2.0  # seconds

# Lyapunov parameters
LYAPUNOV_V = 1.0  # Trade-off parameter
L_MAX = 0.3  # Maximum average transmission latency threshold

# Reward function parameters
ALPHA = 1.0  # Inference delay weight
BETA = 1.0   # Transmission delay weight

# IL parameters
IL_LEARNING_RATE = 1e-3
IL_HIDDEN_SIZE = 128

# RL parameters
PPO_HIDDEN_SIZE = 64
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEF = 0.9
PPO_GAE_LAMBDA = 0.97
PPO_GAMMA = 0.99
PPO_LEARNING_RATE = 1e-4
PPO_BATCH_SIZE = 256
PPO_UPDATE_NUM = 4

# Neural network parameters
GRU_HIDDEN_SIZE = 128
FC_HIDDEN_SIZE = 128
TCN_CHANNELS = 128
TCN_KERNEL_SIZE = 3
TCN_NUM_BLOCKS = 2

# Multi-teacher distillation parameters
DISTILL_ALPHA = 0.5  # Logit distillation weight
DISTILL_BETA = 0.5   # Feature distillation weight
DISTILL_TEMPERATURE = 4.0

# Training parameters
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100
BUFFER_SIZE = 10000

# File paths
MODEL_SAVE_PATH = './models/'
LOG_PATH = './logs/'
TRACE_PATH = './traces/'