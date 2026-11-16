"""
train.py — Mandatory program 2.

Usage:
    python train.py [--lr FLOAT] [--iters INT]

What it does:
    1. Loads mileages and prices from data/data.csv.
    2. Runs batch gradient descent on min-max scaled mileage.
    3. Denormalizes theta back to raw km.
    4. Saves theta0 and theta1 to thetas.json.

After this, run:
    python predict.py
"""

import argparse

from dataset import load_dataset
from gradient_descent import cost_j, fit_linear_regression
from thetas_io import save_thetas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a linear regression model on data/data.csv"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate (alpha). Default: 0.1",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10000,
        help="Number of gradient descent iterations. Default: 10000",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Step 1: Load the dataset (experience E) ---
    print("Loading dataset...")
    mileages, prices, m = load_dataset()
    print(f"  m = {m} examples")
    print(f"  mileage range: {min(mileages):.0f} km - {max(mileages):.0f} km")
    print(f"  price range:   {min(prices):.0f} - {max(prices):.0f}")

    # --- Step 2: Run batch GD with min-max scaling ---
    print(f"\nTraining: alpha={args.lr}, iterations={args.iters}")
    print(f"J before training: {cost_j(mileages, prices, 0.0, 0.0):.2f}")
    theta0, theta1 = fit_linear_regression(
        mileages,
        prices,
        learning_rate=args.lr,
        n_iterations=args.iters,
    )

    # --- Step 3: Report trained thetas ---
    print(f"\nConverged thetas (raw km):")
    print(f"  theta0 = {theta0:.6f}")
    print(f"  theta1 = {theta1:.6f}")
    if theta1 > 0:
        print("  WARNING: theta1 is positive. Check scaling or learning rate.")
    else:
        print("  theta1 is negative (more km = lower price).")

    # --- Step 4: Save to disk ---
    path = save_thetas(theta0, theta1)
    print(f"\nThetas saved to: {path}")
    print("Run 'python predict.py' to estimate a price.")


if __name__ == "__main__":
    main()
