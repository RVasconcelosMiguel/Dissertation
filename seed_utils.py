# === seed_utils.py ===
import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int = 42):
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Optional, helps debugging and determinism
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()  # NEW
