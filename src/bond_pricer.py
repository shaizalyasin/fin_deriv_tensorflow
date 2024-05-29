import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class VasicekModel:
    def __init__(self, alpha, beta, sigma):
        self.alpha = tf.Variable(alpha, dtype=tf.float32, name="alpha")
        self.beta = tf.Variable(beta, dtype=tf.float32, name="beta")
        self.sigma = tf.Variable(sigma, dtype=tf.float32, name="sigma")

    def bond_price(self, face_value, rate, maturity):
        B = (1 - tf.exp(-self.alpha * maturity)) / self.alpha
        A = tf.exp((self.beta - (self.sigma**2) / (2 * self.alpha**2)) * (B - maturity) -
                   (self.sigma**2) * (B**2) / (4 * self.alpha))
        return face_value * A * tf.exp(-B * rate)

    def bond_yield(self, face_value, price, maturity):
        return (face_value / price)**(1 / maturity) - 1

    def simulate_interest_rate_path(self, r0, T, dt, num_paths):
        num_steps = int(T / dt)
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = r0
        for t in range(1, num_steps + 1):
            dW = np.random.normal(scale=np.sqrt(dt), size=num_paths)
            dr = self.alpha * (self.beta - paths[:, t - 1]) * dt + self.sigma * dW
            paths[:, t] = paths[:, t - 1] + dr
        return paths

    def monte_carlo_bond_price(self, face_value, r0, maturity, num_paths=1000, dt=1/252):
        paths = self.simulate_interest_rate_path(r0, maturity, dt, num_paths)
        discount_factors = np.exp(-np.sum(paths[:, 1:] * dt, axis=1))
        bond_prices = face_value * discount_factors
        return np.mean(bond_prices)


def vasicek_calibration(market_data, maturities, initial_guess=(0.1, 0.05, 0.01), learning_rate=0.001, epochs=100):
    alpha, beta, sigma = initial_guess
    model = VasicekModel(alpha, beta, sigma)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def objective_function():
        errors = []
        for i, T in enumerate(maturities):
            price = model.bond_price(1, market_data[i], T)
            model_yield = model.bond_yield(1, price, T)
            market_yield = market_data[i]
            errors.append((model_yield - market_yield) ** 2)
        return tf.reduce_sum(errors)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = objective_function()
        grads = tape.gradient(loss, [model.alpha, model.beta, model.sigma])

        grads = [tf.clip_by_value(grad, -1, 1) for grad in grads if grad is not None]
        optimizer.apply_gradients(zip(grads, [model.alpha, model.beta, model.sigma]))

        model.sigma.assign(tf.clip_by_value(model.sigma, 0.001, np.inf))
        model.alpha.assign(tf.clip_by_value(model.alpha, 0.001, np.inf))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.numpy()}")
            print(f"Alpha: {model.alpha.numpy()}, Beta: {model.beta.numpy()}, Sigma: {model.sigma.numpy()}")

    return model.alpha.numpy(), model.beta.numpy(), model.sigma.numpy()


class HullWhiteModel:
    def __init__(self, a, sigma, theta):
        self.a = tf.Variable(a, dtype=tf.float32)
        self.sigma = tf.Variable(sigma, dtype=tf.float32)
        self.theta = tf.Variable(theta, dtype=tf.float32)

    def zcb_price(self, t, T, r, face_value, theta_interpolated):
        B = (1 - tf.exp(-self.a * (T - t))) / self.a
        A = tf.exp(
            (B - (T - t)) * (theta_interpolated - 0.5 * self.sigma ** 2 / self.a ** 2) - (self.sigma ** 2 * B ** 2) / (
                        4 * self.a))
        return A * tf.exp(-B * r) * face_value

    def get_zcb_price(self, face_value, initial_rate, maturity):
        maturity_tensor = tf.constant([maturity], dtype=tf.float32)
        theta_interpolated = tfp.math.interp_regular_1d_grid(maturity_tensor, 0, 30, self.theta,
                                                             fill_value="extrapolate")
        price = self.zcb_price(0, maturity_tensor, initial_rate, face_value, theta_interpolated)
        return price.numpy()[0]

    def bond_yield(self, price, face_value, maturity):
        return (face_value / price) ** (1 / maturity) - 1

    def monte_carlo_zcb_price(self, face_value, initial_rate, maturity, num_simulations=10000, time_steps=100):
        dt = maturity / time_steps
        rates = np.full((num_simulations,), initial_rate)
        for t in range(1, time_steps + 1):
            theta_interpolated = tfp.math.interp_regular_1d_grid(tf.constant([t * dt], dtype=tf.float32), 0, 30,
                                                                 self.theta, fill_value="extrapolate").numpy()
            rates += self.a.numpy() * (theta_interpolated - rates) * dt + self.sigma.numpy() * np.sqrt(
                dt) * np.random.normal(size=num_simulations)
        prices = face_value * np.exp(-rates * maturity)
        return np.mean(prices)


def hull_white_calibration(rates, maturities, initial_a=0.1, initial_sigma=0.01, learning_rate=0.01, epochs=1000):
    model = HullWhiteModel(initial_a, initial_sigma, tf.zeros_like(rates, dtype=tf.float32))

    time_grid = tf.constant(maturities, dtype=tf.float32)

    def objective_function(model):
        theta_interpolated = tfp.math.interp_regular_1d_grid(time_grid, 0, 30, model.theta, fill_value="extrapolate")
        bond_prices_model = model.zcb_price(0, maturities, rates, 100, theta_interpolated)
        bond_prices_market = tf.exp(-rates * maturities) * 100
        return tf.reduce_mean((bond_prices_model - bond_prices_market) ** 2)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            loss = objective_function(model)
        gradients = tape.gradient(loss, [model.a, model.sigma, model.theta])
        optimizer.apply_gradients(zip(gradients, [model.a, model.sigma, model.theta]))

    return model
