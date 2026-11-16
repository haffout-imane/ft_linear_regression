"""
precision.py — Bonus: measure how well the trained line fits the data.

Usage:
    python train.py        # run first
    python precision.py

Metrics computed (all on the 24 training points — no train/test split):
    MAE   — mean absolute error (average miss in price units)
    MSE   — mean squared error  (large mistakes count more)
    RMSE  — sqrt(MSE), same units as price
    R^2   — fraction of price variance explained by the line (1 = perfect)
"""

import math

from dataset import load_dataset
from gradient_descent import estimate_price
from thetas_io import load_thetas


def mean_absolute_error(mileages, prices, theta0, theta1):
    """
    MAE = (1/m) * sum |estimate - price|

    Average distance between the line and each dot, in price units.
    Easy to say at defense: 'on average we are off by X euros.'
    """
    m = len(mileages)
    total = 0.0
    for i in range(m):
        total += abs(estimate_price(mileages[i], theta0, theta1) - prices[i])
    return total / m


def mean_squared_error(mileages, prices, theta0, theta1):
    """
    MSE = (1/m) * sum (estimate - price)^2

    Same sum as cost J but divided by m, not 2m.
    Large errors are penalised more than small ones (they are squared).
    """
    m = len(mileages)
    total = 0.0
    for i in range(m):
        error = estimate_price(mileages[i], theta0, theta1) - prices[i]
        total += error * error
    return total / m


def root_mean_squared_error(mileages, prices, theta0, theta1):
    """RMSE = sqrt(MSE) — same units as price, easier to interpret than MSE."""
    return math.sqrt(mean_squared_error(mileages, prices, theta0, theta1))


def r_squared(mileages, prices, theta0, theta1):
    """
    R^2 = 1 - SS_res / SS_tot

    SS_res = sum (price - estimate)^2   <- residual sum of squares
    SS_tot = sum (price - mean_price)^2 <- total variance of prices

    R^2 = 1  -> perfect fit
    R^2 = 0  -> no better than always predicting the average price
    R^2 < 0  -> worse than the average (broken model)
    """
    m = len(mileages)
    mean_price = sum(prices) / m

    ss_res = 0.0
    ss_tot = 0.0
    for i in range(m):
        ss_res += (prices[i] - estimate_price(mileages[i], theta0, theta1)) ** 2
        ss_tot += (prices[i] - mean_price) ** 2

    if ss_tot == 0:
        # All prices are identical — R^2 is undefined; model is trivially perfect
        return 1.0

    return 1 - ss_res / ss_tot


def main():
    theta0, theta1 = load_thetas()

    if theta0 == 0.0 and theta1 == 0.0:
        print("WARNING: thetas are 0 -- run 'python train.py' first.")
        return

    mileages, prices, m = load_dataset()

    print(f"theta0 = {theta0:.6f}")
    print(f"theta1 = {theta1:.6f}")
    print(f"m      = {m}\n")

    mae  = mean_absolute_error(mileages, prices, theta0, theta1)
    mse  = mean_squared_error(mileages, prices, theta0, theta1)
    rmse = root_mean_squared_error(mileages, prices, theta0, theta1)
    r2   = r_squared(mileages, prices, theta0, theta1)

    print("--- Precision on training set ---")
    print(f"MAE  : {mae:.2f}   (avg absolute miss, in price units)")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}  (same units as price)")
    print(f"R^2  : {r2:.4f}  (1 = perfect, 0 = no better than avg price)")

    print()
    if r2 >= 0.7:
        print("R^2 >= 0.7 -- the line explains most of the price variance.")
    elif r2 >= 0:
        print("R^2 in [0, 0.7) -- partial fit; noise limits the linear model.")
    else:
        print("R^2 < 0 -- model is worse than predicting the mean. Check training.")

    print()
    print("NOTE: these metrics are on the same 24 points used for training.")
    print("Training error is optimistic -- a real workflow would hold out some cars.")


if __name__ == "__main__":
    main()
