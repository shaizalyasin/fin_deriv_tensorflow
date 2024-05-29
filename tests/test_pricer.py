import pytest
import tensorflow as tf
import numpy as np
from scipy import stats

from src.pricer import BlackScholesPricer, BinomialTreePricer, MonteCarloPricer
from src.contract import EuropeanOptionContract
from src.model import MarketModel
from src.params import Params, TreeParams, MCParams
from src.enums import PutCall


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_fair_value",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 18.14),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 3.75),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 5.13),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 5.65),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 6467.00),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 145.80)
    ]
)
def test_fair_value(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_fair_value):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    pricer = BlackScholesPricer(contract, model, Params())

    assert np.isclose(pricer.calc_fair_value(), expected_fair_value, atol=0.1)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_delta",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 0.7731),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, -0.2269),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 0.5156),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, -0.4843),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 0.9614),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, -0.0385)
    ]
)
def test_delta(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_delta):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    pricer = BlackScholesPricer(contract, model, Params())

    assert np.isclose(pricer.calc_delta(), expected_delta, atol=0.01)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_gamma",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 0.0124),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 0.0124),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 0.0295),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 0.0295),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 0.0000),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 0.0000)
    ]
)
def test_gamma(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_gamma):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    pricer = BlackScholesPricer(contract, model, Params())

    assert np.isclose(pricer.calc_gamma(), expected_gamma, atol=0.01)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_vega",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 30.1940),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 30.1940),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 35.8771),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 35.8771),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 1182.6055),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 1182.6055)
    ]
)
def test_vega(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_vega):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    pricer = BlackScholesPricer(contract, model, Params())

    assert np.isclose(pricer.calc_vega(), expected_vega, atol=0.01)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_theta",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, -6.7287),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, -2.4481),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, -6.8160),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 2.2323),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, -1869.3988),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, -30.0015)
    ]
)
def test_theta(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_theta):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    pricer = BlackScholesPricer(contract, model, Params())

    assert np.isclose(pricer.calc_theta(), expected_theta, atol=0.01)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_rho",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 59.0892),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, -26.5214),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 41.2526),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, -49.2310),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 6294.9897),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, -1062.5992)
    ]
)
def test_rho(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, expected_rho):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    pricer = BlackScholesPricer(contract, model, Params())

    assert np.isclose(pricer.calc_rho(), expected_rho, atol=0.01)


# Binomial Tree Pricer Testing
@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_fair_value",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 1, 19.70),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 1, 5.31),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 10, 5.00),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 10, 5.48),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 200, 6464.71),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 200, 143.55)
    ]
)
def test_fair_value_bt(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps,
                       expected_fair_value):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    tree_param = TreeParams(num_steps=num_steps)
    pricer = BinomialTreePricer(contract, model, tree_param)

    assert np.isclose(pricer.calc_fair_value(), expected_fair_value, atol=0.1)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_delta",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 1, 0.6587),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 1, -0.3412),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 10, 0.4249),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 10, -0.5750),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 200, 0.9673),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 200, -0.0326)
    ]
)
def test_delta_bt(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_delta):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    tree_param = TreeParams(num_steps=num_steps)
    pricer = BinomialTreePricer(contract, model, tree_param)

    assert np.isclose(pricer.calc_delta(), expected_delta, atol=0.01)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_gamma",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 1, 0.0067),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 1, 0.0067),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 10, 0.0291),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 10, 0.0291),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 200, 0.0000),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 200, 0.0000)
    ]
)
def test_gamma_bt(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_gamma):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    tree_param = TreeParams(num_steps=num_steps)
    pricer = BinomialTreePricer(contract, model, tree_param)

    assert np.isclose(pricer.calc_gamma(), expected_gamma, atol=0.01)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_vega",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 1, 41.7437),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 1, 41.7437),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 10, 30.2191),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 10, 30.2191),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 200, 1175.4447),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 200, 1175.45)
    ]
)
def test_vega_bt(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_vega):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    tree_param = TreeParams(num_steps=num_steps)
    pricer = BinomialTreePricer(contract, model, tree_param)

    assert np.isclose(pricer.calc_vega(), expected_vega, atol=0.01)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_theta",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 1, -8.0334),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 1, -3.7529),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 10, -6.2407),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 10, 2.8076),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 200, -1722.0078),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 200, 117.3815)
    ]
)
def test_theta_bt(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_theta):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    tree_param = TreeParams(num_steps=num_steps)
    pricer = BinomialTreePricer(contract, model, tree_param)

    assert np.isclose(pricer.calc_theta(), expected_theta, atol=0.01)


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_rho",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 1, 56.3102),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 1, -29.3004),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 10, 39.7428),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 10, -50.7409),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 200, 6300.3086),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 200, -1057.2511)
    ]
)
def test_rho_bt(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_steps, expected_rho):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    tree_param = TreeParams(num_steps=num_steps)
    pricer = BinomialTreePricer(contract, model, tree_param)

    assert np.isclose(pricer.calc_rho(), expected_rho, atol=0.01)


# Monte Carlo Pricer Testing
@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps, expected_fair_value",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 75, 5, 18.82),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 75, 5, 3.64),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 250, 20, 5.00),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 250, 20, 5.48),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 500, 50, 6464.71),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 500, 50, 143.55)
    ]
)
def test_fair_value_mc(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps,
                       expected_fair_value):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    mc_param = MCParams(num_paths=num_paths, time_steps=time_steps)
    pricer = MonteCarloPricer(contract, model, mc_param)

    # Set the random seed for reproducibility
    tf.random.set_seed(42)
    num_runs = 10
    results = [pricer.calc_fair_value().numpy() for _ in range(num_runs)]

    # Perform t-test
    t_stat, p_value = stats.ttest_1samp(results, expected_fair_value)

    # p-value is greater than 0.05 (95% confidence level)
    assert p_value > 0.05, (f"P-value {p_value} indicates that the result is significantly different "
                            f"from the expected value {expected_fair_value}")
    # Mean and the standard error of the mean
    # mean_result = np.mean(results)
    # sem_result = stats.sem(results)
    # # 95% confidence interval
    # confidence_interval = stats.t.interval(0.95, num_runs - 1, loc=mean_result, scale=sem_result)
    #
    # # Assert that the expected value lies within the confidence interval
    # assert (expected_fair_value >= confidence_interval[0]) and (expected_fair_value <= confidence_interval[1]), \
    #     f"Expected fair value {expected_fair_value} not within confidence interval {confidence_interval}"


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps, expected_delta",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 75, 5, 0.8154),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 75, 5, -0.2162),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 250, 20, 0.5217),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 250, 20, -0.4786),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 500, 50, 0.9744),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 500, 50, -0.0407)
    ]
)
def test_delta_mc(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps,
                  expected_delta):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    mc_param = MCParams(num_paths=num_paths, time_steps=time_steps)
    pricer = MonteCarloPricer(contract, model, mc_param)

    # Set the random seed for reproducibility
    tf.random.set_seed(42)
    num_runs = 10
    results = [pricer.calc_delta() for _ in range(num_runs)]

    t_stat, p_value = stats.ttest_1samp(results, expected_delta)

    # p-value is greater than 0.05 (95% confidence level)
    assert p_value > 0.05, (f"P-value {p_value} indicates that the result is significantly "
                            f"different from the expected value {expected_delta}")
    # Mean and the standard error of the mean
    # mean_result = np.mean(results)
    # sem_result = stats.sem(results)
    # # 95% confidence interval
    # confidence_interval = stats.t.interval(0.95, num_runs - 1, loc=mean_result, scale=sem_result)
    # # Assert that the expected value lies within the confidence interval
    # assert (expected_delta >= confidence_interval[0]) and (expected_delta <= confidence_interval[1]), \
    #     f"Expected delta value {expected_delta} not within confidence interval {confidence_interval}"


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps, expected_gamma",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 75, 5, 0.0103),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 75, 5, 0.0109),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 250, 20, 0.02656583),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 250, 20, 0.0253),
        (PutCall.CALL, 1000, 900, 10.0, 20.0, 0.6, 500, 10, 0.00185546),
        (PutCall.PUT, 1000, 900, 10.0, 20.0, 0.6, 500, 10, 0.00179221)
    ]
)
def test_gamma_mc(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps,
                  expected_gamma):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    mc_param = MCParams(num_paths=num_paths, time_steps=time_steps)
    pricer = MonteCarloPricer(contract, model, mc_param)

    # Set the random seed for reproducibility
    tf.random.set_seed(42)
    num_runs = 10
    results = [pricer.calc_gamma() for _ in range(num_runs)]

    t_stat, p_value = stats.ttest_1samp(results, expected_gamma)

    # p-value is greater than 0.05 (95% confidence level)
    assert p_value > 0.05, (f"P-value {p_value} indicates that the result is significantly "
                            f"different from the expected value {expected_gamma}")

    # mean_result = np.mean(results)
    # sem_result = stats.sem(results)
    # # 95% confidence interval
    # confidence_interval = stats.t.interval(0.95, num_runs - 1, loc=mean_result, scale=sem_result)
    # # Assert that the expected value lies within the confidence interval
    # assert (expected_gamma >= confidence_interval[0]) and (expected_gamma <= confidence_interval[1]), \
    #     f"Expected gamma value {expected_gamma} not within confidence interval {confidence_interval}"


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps, expected_vega",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 75, 5, 32.9011),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 75, 5, 31.8881),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 250, 20, 35.4522),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 250, 20, 34.2712),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 500, 50, 900.7246),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 500, 50, 1161.4688)
    ]
)
def test_vega_mc(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps,
                 expected_vega):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    mc_param = MCParams(num_paths=num_paths, time_steps=time_steps)
    pricer = MonteCarloPricer(contract, model, mc_param)

    # Set the random seed for reproducibility
    tf.random.set_seed(42)
    num_runs = 10
    results = [pricer.calc_vega() for _ in range(num_runs)]

    t_stat, p_value = stats.ttest_1samp(results, expected_vega)

    # p-value is greater than 0.05 (95% confidence level)
    assert p_value > 0.05, (f"P-value {p_value} indicates that the result is significantly "
                            f"different from the expected value {expected_vega}")

    # mean_result = np.mean(results)
    # sem_result = stats.sem(results)
    # # 95% confidence interval
    # confidence_interval = stats.t.interval(0.95, num_runs - 1, loc=mean_result, scale=sem_result)
    # # expected value lies within the confidence interval
    # assert (expected_vega >= confidence_interval[0]) and (expected_vega <= confidence_interval[1]), \
    #     f"Expected vega value {expected_vega} not within confidence interval {confidence_interval}"


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps, expected_theta",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 75, 5, -6.5019),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 75, 5, -2.2042),
        (PutCall.CALL, 90, 120, 10.0, 15.0, 1, 250, 20, -1.9361),
        (PutCall.PUT, 90, 120, 10.0, 15.0, 1, 250, 20, 8.7800),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 500, 50, -1772.5776),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 500, 50, 132.4510)
    ]
)
def test_theta_mc(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps,
                  expected_theta):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    mc_param = MCParams(num_paths=num_paths, time_steps=time_steps)
    pricer = MonteCarloPricer(contract, model, mc_param)

    # Set the random seed for reproducibility
    tf.random.set_seed(42)
    num_runs = 10
    results = [pricer.calc_theta() for _ in range(num_runs)]

    t_stat, p_value = stats.ttest_1samp(results, expected_theta)

    # p-value is greater than 0.05 (95% confidence level)
    assert p_value > 0.05, (f"P-value {p_value} indicates that the result is significantly "
                            f"different from the expected value {expected_theta}")

    # mean_result = np.mean(results)
    # sem_result = stats.sem(results)
    # # 95% confidence interval
    # confidence_interval = stats.t.interval(0.95, num_runs - 1, loc=mean_result, scale=sem_result)
    # # expected value lies within the confidence interval
    # assert (expected_theta >= confidence_interval[0]) and (expected_theta <= confidence_interval[1]), \
    #     f"Expected theta value {expected_theta} not within confidence interval {confidence_interval}"


@pytest.mark.parametrize(
    "derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps, expected_rho",
    [
        (PutCall.CALL, 100, 90, 5.0, 25.0, 1, 750, 50, 59.5967),
        (PutCall.PUT, 100, 90, 5.0, 25.0, 1, 750, 50, -26.2004),
        (PutCall.CALL, 90, 100, 10.0, 15.0, 1, 250, 20, 42.1954),
        (PutCall.PUT, 90, 100, 10.0, 15.0, 1, 250, 20, -48.4528),
        (PutCall.CALL, 10000, 10000, 50.0, 50.0, 2, 500, 50, 6263.0596),
        (PutCall.PUT, 10000, 10000, 50.0, 50.0, 2, 500, 50, -1100.9449)
    ]
)
def test_rho_mc(derivative_type, spot, strike, risk_free_rate, sigma, time_to_expiry, num_paths, time_steps,
                expected_rho):
    contract = EuropeanOptionContract(strike=strike, expiry=time_to_expiry, derivative_type=derivative_type)
    model = MarketModel(spot=spot, risk_free_rate=risk_free_rate, sigma=sigma)
    mc_param = MCParams(num_paths=num_paths, time_steps=time_steps)
    pricer = MonteCarloPricer(contract, model, mc_param)

# Set the random seed for reproducibility
    tf.random.set_seed(42)
    num_runs = 10
    results = [pricer.calc_rho() for _ in range(num_runs)]

    t_stat, p_value = stats.ttest_1samp(results, expected_rho)

    # p-value is greater than 0.05 (95% confidence level)
    assert p_value > 0.05, (f"P-value {p_value} indicates that the result is significantly "
                            f"different from the expected value {expected_rho}")

    # mean_result = np.mean(results)
    # sem_result = stats.sem(results)
    # # 95% confidence interval
    # confidence_interval = stats.t.interval(0.95, num_runs - 1, loc=mean_result, scale=sem_result)
    # # expected value lies within the confidence interval
    # assert (expected_rho >= confidence_interval[0]) and (expected_rho <= confidence_interval[1]), \
    #     f"Expected rho value {expected_rho} not within confidence interval {confidence_interval}"
