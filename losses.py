import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.75, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        loss = -self.alpha * tf.pow(1. - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)

    def get_config(self):
        # Only pure Python types allowed here
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
