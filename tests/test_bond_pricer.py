import pytest
import numpy as np
import tensorflow as tf

from src.bond_pricer import VasicekModel, vasicek_calibration, HullWhiteModel


# Vasicek Model Testing
@pytest.fixture
def sample_vasicek_model():
    return VasicekModel(alpha=0.1, beta=0.05, sigma=0.02)


@pytest.mark.parametrize("alpha, beta, sigma", [
    (0.2, 0.05, 0.01),  # Original test case
    (0.1, 0.03, 0.02),  # Different set of lower values
    (0.5, 0.1, 0.05),   # Higher values
    (0.01, 0.005, 0.001)  # Very low values
])
def test_initialization(alpha, beta, sigma):
    model = VasicekModel(alpha, beta, sigma)
    assert np.isclose(model.alpha.numpy(), alpha, atol=1e-6), f"Alpha expected {alpha}, got {model.alpha.numpy()}"
    assert np.isclose(model.beta.numpy(), beta, atol=1e-6), f"Beta expected {beta}, got {model.beta.numpy()}"
    assert np.isclose(model.sigma.numpy(), sigma, atol=1e-6), f"Sigma expected {sigma}, got {model.sigma.numpy()}"


@pytest.mark.parametrize("face_value, rate, maturity, expected_price", [
    (100, 0.05, 5, 78.33),
    (100, 0.03, 3, 90.78),
    (100, 0.07, 7, 63.91),
    (100, 0.04, 1, 96.05)
])
def test_bond_price(sample_vasicek_model, face_value, rate, maturity, expected_price):
    price = sample_vasicek_model.bond_price(face_value, rate, maturity)
    assert np.isclose(price, expected_price, atol=1)


@pytest.mark.parametrize("face_value, price, maturity, expected_yield", [
    (100, 95, 5, 0.0107),
    (100, 90, 10, 0.0108),
    (100, 105, 2, -0.0247),
    (100, 100, 3, 0.0000)
])
def test_bond_yield(sample_vasicek_model, face_value, price, maturity, expected_yield):
    yield_calc = sample_vasicek_model.bond_yield(face_value, price, maturity)
    assert np.isclose(yield_calc, expected_yield, atol=1e-3)


@pytest.mark.parametrize("r0, T, dt, num_paths", [
    (0.05, 1, 1/252, 10),
    (0.03, 1, 1/365, 5),
    (0.04, 0.5, 1/252, 20),
    (0.06, 2, 1/252, 15)
])
def test_simulate_interest_rate_path(sample_vasicek_model, r0, T, dt, num_paths):
    paths = sample_vasicek_model.simulate_interest_rate_path(r0, T, dt, num_paths)
    assert paths.shape == (num_paths, int(T/dt) + 1)
    assert np.allclose(paths[:, 0], r0, atol=1e-6), "Initial rates are not as expected"


@pytest.mark.parametrize("face_value, r0, maturity, num_paths, expected_price", [
    (100, 0.05, 1, 100, 95.24),
    (100, 0.03, 2, 50, 94.34),
    (100, 0.07, 1, 150, 93.36),
    (100, 0.04, 0.5, 200, 98.02)
])
def test_monte_carlo_bond_price(sample_vasicek_model, face_value, r0, maturity, num_paths, expected_price):
    price = sample_vasicek_model.monte_carlo_bond_price(face_value, r0, maturity, num_paths=num_paths)
    assert np.isclose(price, expected_price, atol=2)


def test_vasicek_calibration():
    market_data = [0.03, 0.035, 0.04]
    maturities = [1, 2, 3]
    alpha, beta, sigma = vasicek_calibration(market_data, maturities, epochs=10)
    assert alpha > 0 and beta > 0 and sigma > 0


# Hull White Testing
@pytest.mark.parametrize("a, sigma, theta", [
    (0.1, 0.01, 0.02),
    (0.2, 0.02, 0.03),
    (0.05, 0.005, 0.01),
    (0.3, 0.03, 0.04)
])
def test_initialization(a, sigma, theta):
    model = HullWhiteModel(a, sigma, theta)
    assert np.isclose(model.a.numpy(), a)
    assert np.isclose(model.sigma.numpy(), sigma)
    assert np.isclose(model.theta.numpy(), theta)


@pytest.mark.parametrize("a, sigma, theta, t, T, r, face_value, expected_price", [
    (0.1, 0.01, 0.02, 0, 5, 0.05, 100, 80.52),
    (0.2, 0.02, 0.025, 0, 10, 0.03, 100, 77.67),
    (0.15, 0.015, 0.015, 0, 2, 0.04, 100, 92.96),
    (0.25, 0.03, 0.03, 0, 8, 0.06, 100, 72.48)
])
def test_zcb_price(a, sigma, theta, t, T, r, face_value, expected_price):
    model = HullWhiteModel(a, sigma, theta)
    theta_interpolated = tf.constant([theta], dtype=tf.float32)
    price = model.zcb_price(t, T, r, face_value, theta_interpolated).numpy()
    assert np.isclose(price, expected_price, atol=1)


@pytest.mark.parametrize("price, face_value, maturity, expected_yield", [
    (95, 100, 5, 0.0103),
    (90, 100, 10, 0.0106),
    (105, 100, 2, -0.0241),
    (100, 100, 3, 0.0000)
])
def test_bond_yield(price, face_value, maturity, expected_yield):
    model = HullWhiteModel(0.1, 0.01, 0.02)
    yield_calc = model.bond_yield(price, face_value, maturity)
    assert np.isclose(yield_calc, expected_yield, atol=1e-3)
