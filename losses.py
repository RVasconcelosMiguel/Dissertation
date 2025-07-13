# === losses.py ===
import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=2.0, alpha=None):
    """
    Focal Loss for binary classification.
    gamma: focusing parameter.
    alpha: class balancing weight (float in [0, 1] or None).
           If alpha is None, no class weighting is applied.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

        if alpha is not None:
            weight = alpha * K.pow(1 - y_pred, gamma) * y_true + \
                     (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true)
        else:
            weight = K.pow(1 - y_pred, gamma) * y_true + \
                     K.pow(y_pred, gamma) * (1 - y_true)

        loss = weight * cross_entropy
        return K.mean(loss)

    return focal_loss_fixed
