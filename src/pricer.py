from abc import ABC, abstractmethod
from src.contract import EuropeanOptionContract
from src.model import MarketModel
from src.params import Params, MCParams, TreeParams
from src.enums import PutCall
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Pricer(ABC):
    def __init__(self, contract: EuropeanOptionContract, model: MarketModel, params: Params) -> None:
        self.contract: EuropeanOptionContract = contract
        self.model: MarketModel = model
        self.params: Params | MCParams | TreeParams = params

    @abstractmethod
    def calc_fair_value(self):
        pass


class BlackScholesPricer(Pricer):
    def __init__(self, contract: EuropeanOptionContract, model: MarketModel, params: Params) -> None:
        super().__init__(contract, model, params)

    @staticmethod
    def calc_d1(spot, strike, risk_free_rate, sigma, time_to_expiry):
        d1 = ((tf.math.log(spot / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_expiry) /
              (sigma * tf.sqrt(time_to_expiry)))
        return d1

    @staticmethod
    def calc_d2(spot, strike, risk_free_rate, sigma, time_to_expiry):
        d1 = BlackScholesPricer.calc_d1(spot, strike, risk_free_rate, sigma, time_to_expiry)
        d2 = d1 - sigma * tf.sqrt(time_to_expiry)
        return d2

    def calc_fair_value(self):
        """ Fair value of an option using the Black-Scholes formula based on current market conditions
        and contract specifics."""
        spot = self.model.spot
        risk_free_rate = self.model.risk_free_rate
        sigma = self.model.sigma
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry

        d1 = BlackScholesPricer.calc_d1(spot, strike, risk_free_rate, sigma, time_to_expiry)
        d2 = BlackScholesPricer.calc_d2(spot, strike, risk_free_rate, sigma, time_to_expiry)
        print(f"D1 = {d1}")
        print(f"D2 = {d2}")

        if self.contract.derivative_type == PutCall.CALL:
            return spot * tfp.distributions.Normal(0., 1.).cdf(d1) - strike * tf.exp(-risk_free_rate * time_to_expiry)\
                    * tfp.distributions.Normal(0., 1.).cdf(d2)
        elif self.contract.derivative_type == PutCall.PUT:
            return strike * tf.exp(-risk_free_rate * time_to_expiry) * tfp.distributions.Normal(0., 1.).cdf(-d2) - \
                    spot * tfp.distributions.Normal(0., 1.).cdf(-d1)

    def calc_delta(self):
        """Rate of change of the option's price with respect to changes in the price of the underlying asset."""
        spot = self.model.spot
        risk_free_rate = self.model.risk_free_rate
        sigma = self.model.sigma
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry

        d1 = BlackScholesPricer.calc_d1(spot, strike, risk_free_rate, sigma, time_to_expiry)

        if self.contract.derivative_type == PutCall.CALL:
            return tfp.distributions.Normal(0., 1.).cdf(d1)
        elif self.contract.derivative_type == PutCall.PUT:
            return tfp.distributions.Normal(0., 1.).cdf(d1) - 1

    def calc_gamma(self):
        """Rate of change of Delta with respect to changes in the price of the underlying asset."""
        spot = self.model.spot
        risk_free_rate = self.model.risk_free_rate
        sigma = self.model.sigma
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry

        d1 = BlackScholesPricer.calc_d1(spot, strike, risk_free_rate, sigma, time_to_expiry)
        normal_dist = tfp.distributions.Normal(0., 1.)
        pdf_d1 = normal_dist.prob(d1)

        return pdf_d1 / (spot * sigma * tf.sqrt(time_to_expiry))

    def calc_vega(self):
        """Rate of change of the option's price with respect to changes in the volatility of the underlying asset."""
        spot = self.model.spot
        risk_free_rate = self.model.risk_free_rate
        sigma = self.model.sigma
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry

        d1 = BlackScholesPricer.calc_d1(spot, strike, risk_free_rate, sigma, time_to_expiry)
        pdf_d1 = tfp.distributions.Normal(0., 1.).prob(d1)

        return spot * tf.sqrt(time_to_expiry) * pdf_d1

    def calc_theta(self):
        """Rate of change of the option's price with respect to the passage of time."""
        spot = self.model.spot
        risk_free_rate = self.model.risk_free_rate
        sigma = self.model.sigma
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry

        d1 = BlackScholesPricer.calc_d1(spot, strike, risk_free_rate, sigma, time_to_expiry)
        pdf_d1 = tfp.distributions.Normal(0., 1.).prob(d1)
        d2 = BlackScholesPricer.calc_d2(spot, strike, risk_free_rate, sigma, time_to_expiry)

        if self.contract.derivative_type == PutCall.CALL:
            return (-spot * pdf_d1 * sigma / 2 * tf.sqrt(time_to_expiry) - risk_free_rate * strike *
                    tf.exp(-risk_free_rate * time_to_expiry) * tfp.distributions.Normal(0., 1.).cdf(d2))
        elif self.contract.derivative_type == PutCall.PUT:
            return (-spot * pdf_d1 * sigma / 2 * tf.sqrt(time_to_expiry) + risk_free_rate * strike *
                    tf.exp(-risk_free_rate * time_to_expiry) * tfp.distributions.Normal(0., 1.).cdf(-d2))

    def calc_rho(self):
        """Rate of change of the option's price with respect to changes in the risk-free interest rate."""
        spot = self.model.spot
        risk_free_rate = self.model.risk_free_rate
        sigma = self.model.sigma
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry

        d2 = BlackScholesPricer.calc_d2(spot, strike, risk_free_rate, sigma, time_to_expiry)

        if self.contract.derivative_type == PutCall.CALL:
            return (strike * time_to_expiry * tf.exp(-risk_free_rate * time_to_expiry) *
                    tfp.distributions.Normal(0., 1.).cdf(d2))
        elif self.contract.derivative_type == PutCall.PUT:
            return (-1.0 * strike * time_to_expiry * tf.exp(-risk_free_rate * time_to_expiry) *
                    tfp.distributions.Normal(0., 1.).cdf(-d2))


class BinomialTreePricer(Pricer):
    def __init__(self, contract: EuropeanOptionContract, model: MarketModel, params: TreeParams) -> None:
        super().__init__(contract, model, params)
        self.num_steps = params.num_steps  # Number of steps in the binomial tree
        self.stock_tree = None
        self.option_tree = None

    def calc_fair_value(self):
        """Fair value of an option using the Binomial Tree method based on current market conditions
        and european contract specifics."""
        spot = self.model.spot
        risk_free_rate = self.model.risk_free_rate
        sigma = self.model.sigma
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry
        num_steps = self.num_steps

        dt = time_to_expiry / num_steps  # Time step
        u = tf.exp(sigma * tf.sqrt(dt))  # Upward factor
        d = 1 / u  # Downward factor
        q = (tf.exp(risk_free_rate * dt) - d) / (u - d)  # risk-neutral probability

        up_range = tf.cast(tf.range(num_steps, -1, -1), dtype=tf.float32)
        down_range = tf.cast(tf.range(0, num_steps + 1), dtype=tf.float32)

        stock_prices = spot * d ** up_range * u ** down_range
        option_values = tf.maximum(0.0,
                                   stock_prices - strike if self.contract.derivative_type == PutCall.CALL else
                                   strike - stock_prices)

        for i in range(num_steps - 1, -1, -1):
            option_values = (q * option_values[1:] + (1 - q) * option_values[:-1]) * tf.exp(-risk_free_rate * dt)

        return option_values[0]

    def reset_tree(self):
        """Reset the stock and option trees to initial states."""
        self.stock_tree = np.zeros((self.num_steps + 1, self.num_steps + 1))
        self.option_tree = np.zeros((self.num_steps + 1, self.num_steps + 1))

    def build_tree(self):
        spot = self.model.spot.numpy()
        risk_free_rate = self.model.risk_free_rate.numpy()
        sigma = self.model.sigma.numpy()
        strike = self.contract.strike.numpy()
        time_to_expiry = self.contract.expiry.numpy()
        num_steps = self.num_steps

        dt = time_to_expiry / num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(risk_free_rate * dt) - d) / (u - d)

        self.reset_tree()

        for i in range(num_steps + 1):
            for j in range(i + 1):
                self.stock_tree[j, i] = spot * (u ** (i - j)) * (d ** j)

        for j in range(num_steps + 1):
            self.option_tree[j, num_steps] = max(0, self.stock_tree[j, num_steps] - strike) \
                if (self.contract.derivative_type == PutCall.CALL) else max(0, strike - self.stock_tree[j, num_steps])

        for i in range(num_steps - 1, -1, -1):
            for j in range(i + 1):
                self.option_tree[j, i] = np.exp(-risk_free_rate * dt) * (q * self.option_tree[j, i + 1] + (1 - q) *
                                                                         self.option_tree[j + 1, i + 1])

        print("Stock Tree:\n", self.stock_tree)
        print("Option Tree:\n", self.option_tree)

    def calc_delta(self):

        spot = self.model.spot
        with tf.GradientTape() as tape:
            tape.watch(spot)
            option_price = self.calc_fair_value()

        delta = tape.gradient(option_price, spot)
        return delta.numpy()

    def calc_gamma(self):
        """Gamma using finite differences."""
        dt = self.contract.expiry.numpy() / self.num_steps
        sigma = self.model.sigma.numpy()

        if not isinstance(self.model.spot, tf.Variable):
            self.model.spot = tf.Variable(self.model.spot, dtype=tf.float32)

        original_spot = self.model.spot.numpy()

        spot_up = original_spot * np.exp(sigma * np.sqrt(dt))
        spot_down = original_spot * np.exp(-sigma * np.sqrt(dt))

        self.model.spot.assign(spot_up)
        delta_up = self.calc_delta()

        self.model.spot.assign(spot_down)
        delta_down = self.calc_delta()

        gamma = (delta_up - delta_down) / (spot_up - spot_down)
        return gamma

    def calc_vega(self):

        sigma = self.model.sigma
        with tf.GradientTape() as tape:
            tape.watch(sigma)
            option_price = self.calc_fair_value()

        vega = tape.gradient(option_price, sigma)
        return vega.numpy()

    def calc_theta(self):

        time_to_expiry = self.contract.expiry
        with tf.GradientTape() as tape:
            tape.watch(time_to_expiry)
            option_price = self.calc_fair_value()

        # Note the negative sign, as time decay usually reduces the option value
        theta = -tape.gradient(option_price, time_to_expiry)
        return theta.numpy()

    def calc_rho(self):
        risk_free_rate = self.model.risk_free_rate
        with tf.GradientTape() as tape:
            tape.watch(risk_free_rate)
            option_price = self.calc_fair_value()

        rho = tape.gradient(option_price, risk_free_rate)
        return rho.numpy()


class MonteCarloPricer(Pricer):
    def __init__(self, contract: EuropeanOptionContract, model: MarketModel, params: MCParams, save_paths=False):
        super().__init__(contract, model, params)
        self.num_paths = params.num_paths
        self.time_steps = params.time_steps
        self.save_paths = save_paths
        self.paths = None

    def calc_fair_value(self):
        """Fair value of an option using the Monte Carlo method based on current market conditions
        and european contract specifics."""
        spot = self.model.spot
        risk_free_rate = self.model.risk_free_rate
        sigma = self.model.sigma
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry
        num_paths = self.num_paths
        time_steps = self.time_steps

        dt = time_to_expiry / tf.cast(time_steps, tf.float32)

        drift = (risk_free_rate - 0.5 * sigma ** 2) * dt
        shock = sigma * tf.sqrt(dt)

        random_draws = tf.random.normal(shape=(time_steps, num_paths), mean=0.0, stddev=1.0, dtype=tf.float32, seed=42)
        price_paths = spot * tf.exp(tf.cumsum(drift + shock * random_draws, axis=0))

        if self.save_paths:
            self.paths = price_paths.numpy()

        if self.contract.derivative_type == PutCall.CALL:
            payoffs = tf.maximum(price_paths[-1, :] - strike, 0.0)
        elif self.contract.derivative_type == PutCall.PUT:
            payoffs = tf.maximum(strike - price_paths[-1, :], 0.0)
        else:
            raise ValueError("Unsupported option type")

        option_price = tf.exp(-risk_free_rate * time_to_expiry) * tf.reduce_mean(payoffs)

        return option_price

    def get_paths(self):
        return self.paths if self.paths is not None else None

    def calc_delta(self):

        spot = self.model.spot
        with tf.GradientTape() as tape:
            tape.watch(spot)
            option_price = self.calc_fair_value()

        delta = tape.gradient(option_price, spot)
        return delta.numpy()

    def calc_gamma(self):
        """Gamma using finite differences."""
        dt = self.contract.expiry.numpy() / self.time_steps
        sigma = self.model.sigma.numpy()

        if not isinstance(self.model.spot, tf.Variable):
            self.model.spot = tf.Variable(self.model.spot, dtype=tf.float32)

        original_spot = self.model.spot.numpy()
        spot_up = original_spot * np.exp(sigma * np.sqrt(dt))
        spot_down = original_spot * np.exp(-sigma * np.sqrt(dt))

        self.model.spot.assign(spot_up)
        delta_up = self.calc_delta()

        self.model.spot.assign(spot_down)
        delta_down = self.calc_delta()

        gamma = (delta_up - delta_down) / (spot_up - spot_down)
        return gamma

    def calc_vega(self):

        sigma = self.model.sigma
        with tf.GradientTape() as tape:
            tape.watch(sigma)
            option_price = self.calc_fair_value()

        vega = tape.gradient(option_price, sigma)
        return vega.numpy()

    def calc_theta(self):

        time_to_expiry = self.contract.expiry
        with tf.GradientTape() as tape:
            tape.watch(time_to_expiry)
            option_price = self.calc_fair_value()

        theta = -tape.gradient(option_price, time_to_expiry)
        return theta.numpy()

    def calc_rho(self):
        risk_free_rate = self.model.risk_free_rate
        with tf.GradientTape() as tape:
            tape.watch(risk_free_rate)
            option_price = self.calc_fair_value()

        rho = tape.gradient(option_price, risk_free_rate)
        return rho.numpy()
