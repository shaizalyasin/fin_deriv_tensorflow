import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import tensorflow as tf


# Function to convert expiry date to time to expiration in years
def calculate_time_to_expiry(expiry_date):
    if expiry_date:
        expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d')
        today = datetime.today()
        return (expiry_date - today).days / 365.25
    return 0


def plot_convergence(tree_steps, binomial_prices, bs_price):
    """
    Plot the convergence of Binomial Tree model prices to the Black-Scholes price.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(tree_steps, binomial_prices, label='Binomial Tree', marker='o')
    plt.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes')

    plt.title('Convergence of Binomial Tree Model to Black-Scholes (0.7724)')
    plt.xlabel('Number of Tree Steps')
    plt.ylabel('Delta')
    plt.legend()
    plt.grid(True)
    plt.show()


# tree_steps = [30, 60, 90, 120, 140, 170, 200, 250, 300]
# binomial_prices = [0.8087, 0.7618, 0.8028, 0.7825, 0.7725, 0.7605, 0.7510, 0.7785, 0.7668]
# bs_price = 0.7724
# plot_convergence(tree_steps, binomial_prices, bs_price)


def plot_monte_carlo_convergence(num_simulations, monte_carlo_prices, bs_price):
    """
    Plot the convergence of Monte Carlo simulation prices to the Black-Scholes price.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(num_simulations, monte_carlo_prices, label='Monte Carlo Prices', marker='o')
    plt.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes Price')

    plt.title('Convergence of Monte Carlo Simulation Prices to Black-Scholes Price')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# num_simulations = [1000, 5000, 10000, 13000, 15000, 17000, 20000]
# monte_carlo_prices = [16.93, 17.87, 17.90, 17.98, 18.17, 18.10, 18.27]
# bs_price = 18.10
# plot_monte_carlo_convergence(num_simulations, monte_carlo_prices, bs_price)


def simulate_option_payoff(S, K, T, r, sigma, paths):
    dt = T
    Z = np.random.standard_normal(paths)
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(S_T - K, 0) * np.exp(-r * T)
    return payoff


def monte_carlo_error_bounds(S, K, T, r, sigma, total_simulations, chunk_size):
    results = []
    cum_payoff = 0
    cum_square_payoff = 0

    for i in range(0, total_simulations, chunk_size):
        payoffs = simulate_option_payoff(S, K, T, r, sigma, chunk_size)
        cum_payoff += np.sum(payoffs)
        cum_square_payoff += np.sum(payoffs ** 2)

        mean = cum_payoff / (i + chunk_size)
        variance = (cum_square_payoff / (i + chunk_size)) - (mean ** 2)
        std_dev = np.sqrt(variance)
        error_bound = 1.96 * std_dev / np.sqrt(i + chunk_size)

        results.append((mean, error_bound))

    return results


# # Example parameters
# S = 100
# K = 90
# T = 1
# r = 0.05
# sigma = 0.25
# total_simulations = 120000
# chunk_size = 1000
# price = simulate_option_payoff(S, K, T, r, sigma, total_simulations)
# print(f"Option Price = {price}")


# results = monte_carlo_error_bounds(S, K, T, r, sigma, total_simulations, chunk_size)
# for idx, (mean, error) in enumerate(results):
#     print(f"After {(idx + 1) * chunk_size} simulations: Mean = {mean:.2f}, Error Bound = {error:.2f}")


# sim_counts = [chunk_size * (i + 1) for i in range(len(results))]
# means = [res[0] for res in results]
# errors = [res[1] for res in results]


# plt.figure(figsize=(10, 5))
# plt.plot(sim_counts, means, label='Estimated Option Price', color='blue')
# plt.fill_between(sim_counts, [m - e for m, e in zip(means, errors)],
#                  [m + e for m, e in zip(means, errors)], color='gray', alpha=0.5, label='95% Confidence Interval')
# plt.title('Convergence of Monte Carlo Estimates')
# plt.xlabel('Number of Simulations')
# plt.ylabel('Option Price')
# plt.legend()
# plt.grid(True)
# plt.show()


def simulate_greeks(S, K, T, r, sigma, paths):
    # Setup for gradient computation
    S = tf.Variable(float(S), dtype=tf.float32)
    K = tf.constant(float(K), dtype=tf.float32)
    T = tf.constant(float(T), dtype=tf.float32)
    r = tf.constant(float(r), dtype=tf.float32)
    sigma = tf.Variable(float(sigma), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([S, sigma])
        dt = T
        Z = tf.random.normal([paths], dtype=tf.float32)
        S_T = S * tf.exp((r - 0.5 * sigma ** 2) * dt + sigma * tf.sqrt(dt) * Z)
        payoff = tf.maximum(S_T - K, 0) * tf.exp(-r * T)
        price = tf.reduce_mean(payoff)

    delta = tape.gradient(price, S)
    vega = tape.gradient(price, sigma)

    return delta.numpy(), vega.numpy()


def monte_carlo_greeks_error_bounds(S, K, T, r, sigma, total_simulations, chunk_size):
    deltas = []
    vegas = []
    mean_deltas = []  # To store mean delta after each chunk
    error_bounds_deltas = []  # To store error bounds for deltas
    mean_vegas = []  # To store mean vega after each chunk
    error_bounds_vegas = []  # To store error bounds for vegas

    for i in range(0, total_simulations, chunk_size):
        chunk_deltas, chunk_vegas = [], []
        for _ in range(chunk_size):
            delta, vega = simulate_greeks(S, K, T, r, sigma, 1)
            chunk_deltas.append(delta)
            chunk_vegas.append(vega)
        deltas.extend(chunk_deltas)
        vegas.extend(chunk_vegas)

        # cumulative mean
        current_mean_delta = np.mean(deltas)
        current_mean_vega = np.mean(vegas)
        # standard deviation
        std_delta = np.std(deltas)
        std_vega = np.std(vegas)
        # error bound
        current_error_bound_delta = 1.96 * std_delta / np.sqrt(len(deltas))
        current_error_bound_vega = 1.96 * std_vega / np.sqrt(len(vegas))

        mean_deltas.append(current_mean_delta)
        error_bounds_deltas.append(current_error_bound_delta)
        mean_vegas.append(current_mean_vega)
        error_bounds_vegas.append(current_error_bound_vega)

        print(f"After {i + chunk_size} simulations: Delta = {current_mean_delta:.4f}, Error Bound = "
              f"{current_error_bound_delta:.4f}")
        print(f"After {i + chunk_size} simulations: Vega = {current_mean_vega:.4f}, Error Bound = "
              f"{current_error_bound_vega:.4f}")

    return mean_deltas, error_bounds_deltas, mean_vegas, error_bounds_vegas


# mean_deltas, error_bounds_deltas, mean_vegas, error_bounds_vegas = (
#     monte_carlo_greeks_error_bounds(S, K, T, r, sigma, total_simulations, chunk_size))
# simulation_counts = np.arange(1000, total_simulations + 1, 1000)

# Plot for Delta
# plt.figure(figsize=(10, 6))
# plt.plot(simulation_counts, mean_deltas, label='Estimated Delta', color='blue')
# plt.fill_between(simulation_counts, [m - e for m, e in zip(mean_deltas, error_bounds_deltas)],
#                  [m + e for m, e in zip(mean_deltas, error_bounds_deltas)], color='grey', alpha=0.5,
#                  label='95% Confidence Interval for Delta')
# plt.title('Convergence of Delta Estimates')
# plt.xlabel('Number of Simulations')
# plt.ylabel('Delta')
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot for Vega
# plt.figure(figsize=(10, 6))
# plt.plot(simulation_counts, mean_vegas, label='Estimated Vega', color='royalblue')
# plt.fill_between(simulation_counts, [m - e for m, e in zip(mean_vegas, error_bounds_vegas)],
#                  [m + e for m, e in zip(mean_vegas, error_bounds_vegas)], color='lightsteelblue', alpha=0.5,
#                  label='95% Confidence Interval for Vega')
# plt.title('Convergence of Vega Estimates')
# plt.xlabel('Number of Simulations')
# plt.ylabel('Vega')
# plt.legend()
# plt.grid(True)
# plt.show()


def plot_pricing_results(fair_values, deltas, gammas, vegas, thetas, rhos, model_names):
    """
    Plots comparison of scalar values (e.g., Fair Value, Greeks) across different models.
    """
    metrics = [fair_values, deltas, gammas, vegas, thetas, rhos]
    metric_names = ['Fair Value', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']

    fig, axs = plt.subplots(1, len(metric_names), figsize=(20, 5))

    plt.subplots_adjust(wspace=0.4)

    for i, metric in enumerate(metric_names):
        for j, value in enumerate(metrics[i]):
            axs[i].scatter(j, value, label=f'{model_names[j]}: {value:.2f}')
            axs[i].set_xticks(range(len(model_names)))
            axs[i].set_xticklabels(model_names)
        axs[i].set_title(metric)
        axs[i].legend()

    plt.show()


def plot_scalar_comparison(values, labels, title):
    """
    Plots a comparison of scalar values (e.g., fair value) across different models.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['blue', 'orange', 'green'])
    plt.title(title)
    plt.show()


def plot_grouped_comparison(metrics, model_names, metric_names):
    """
    Plots grouped comparison of scalar values (e.g., Fair Value, Greeks) across different models.
    """
    num_metrics = len(metric_names)
    num_models = len(model_names)
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    indices = np.arange(num_metrics)

    for i, model_name in enumerate(model_names):
        model_values = [metric[i] for metric in metrics]
        pos = indices - (num_models / 2 - i) * width
        ax.bar(pos, model_values, width, label=model_name)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Pricing Model Metrics')
    ax.set_xticks(indices)
    ax.set_xticklabels(metric_names)
    ax.legend()

    plt.show()


def plot_scalar_comparison_last(metrics, model_names, metric_names):
    """
    Plots a line comparison of scalar values (e.g., Fair Value, Greeks) across different models.
    """
    num_models = len(model_names)
    x = range(num_models)  # X-axis points representing each model

    for metric_values, metric_name in zip(metrics, metric_names):
        plt.plot(x, metric_values, marker='o', label=metric_name)

    plt.xticks(x, model_names)
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('Comparison of Pricing Model Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_metrics_separately(metrics, model_names, metric_names):
    """
    Plots each metric in a separate graph for comparison across different models.
    """
    fig_size = (8, 4 * len(metric_names))
    plt.figure(figsize=fig_size)

    for i, metric_name in enumerate(metric_names):
        plt.subplot(len(metric_names), 1, i + 1)
        for j, model_name in enumerate(model_names):
            values = [metric[j] for metric in metrics]
            plt.plot(values, '-o', label=model_name)
        plt.title(metric_name)
        plt.xticks(range(len(metric_names)), metric_names)
        plt.legend()
        plt.tight_layout()

    plt.show()


def plot_interest_rates(rates):
    plt.figure(figsize=(10, 6))
    plt.plot(rates, label='Interest Rates')
    plt.title('Simulated Interest Rates')
    plt.xlabel('Time Step')
    plt.ylabel('Rate')
    plt.legend()
    plt.show()
