from abc import ABC, abstractmethod
import tensorflow as tf
from src.enums import PutCall


class Contract(ABC):
    def __init__(self, derivative_type: PutCall, strike: float, expiry: float) -> None:
        self.derivative_type: PutCall = derivative_type
        self.strike = tf.Variable(strike, dtype=tf.float32)
        self.expiry = tf.Variable(expiry, dtype=tf.float32)

    @abstractmethod
    def payoff(self, spot: float) -> tf.Tensor:
        """Abstract method to be implemented by subclasses."""
        pass


class EuropeanOptionContract(Contract):

    def payoff(self, spot: float) -> tf.Tensor:
        """Payoff for a European option contract based on the current spot price."""
        spot_price = tf.constant(spot, dtype=tf.float32)
        if self.derivative_type == PutCall.CALL:
            return tf.maximum(0.0, spot_price - self.strike)
        elif self.derivative_type == PutCall.PUT:
            return tf.maximum(0.0, self.strike - spot_price)
