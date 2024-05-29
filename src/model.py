import tensorflow as tf
from abc import ABC


class MarketModel(ABC):
    def __init__(self, risk_free_rate: float, spot: float, sigma: float):
        self.risk_free_rate = tf.Variable(risk_free_rate / 100.0, dtype=tf.float32)
        self.spot = tf.Variable(spot, dtype=tf.float32)
        self.sigma = tf.Variable(sigma / 100.0, dtype=tf.float32)
