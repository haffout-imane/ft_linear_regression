"""
plot.py — Bonus: scatter of the 24 cars + the fitted regression line.

Usage:
    python train.py          # must run first to have thetas.json
    python plot.py

What it shows:
    - Each car as a dot (km, price).
    - The regression line from xmin to xmax of the dataset.
    - A vertical annotation for the example prediction at 100 000 km.
"""

import matplotlib.pyplot as plt

from dataset import load_dataset
from hypothesis import estimate_price
from thetas_io import load_thetas


def plot_regression(mileages, prices, theta0, theta1):
    """Draw scatter + line + optional prediction marker."""

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # --- Scatter: each car is a dot ---
    ax.scatter(
        mileages,
        prices,
        color="#4fc3f7",
        edgecolors="#ffffff",
        linewidths=0.4,
        s=60,
        zorder=3,
        label="Training data (24 cars)",
    )

    # --- Regression line from xmin to xmax ---
    x_line = [min(mileages), max(mileages)]
    y_line = [estimate_price(x, theta0, theta1) for x in x_line]
    ax.plot(
        x_line,
        y_line,
        color="#ff6b6b",
        linewidth=2,
        zorder=2,
        label=f"h(x) = {theta0:.0f} + {theta1:.5f} * km",
    )

    # --- Example prediction at 100 000 km ---
    example_km = 100_000
    example_price = estimate_price(example_km, theta0, theta1)
    ax.axvline(x=example_km, color="#ffd166", linewidth=1, linestyle="--", alpha=0.6)
    ax.axhline(y=example_price, color="#ffd166", linewidth=1, linestyle="--", alpha=0.6)
    ax.scatter(
        [example_km],
        [example_price],
        color="#ffd166",
        s=100,
        zorder=4,
        label=f"Prediction at 100 000 km: {example_price:.0f}",
    )

    # --- Labels and style ---
    ax.set_xlabel("Mileage (km)", color="#cccccc", fontsize=11)
    ax.set_ylabel("Price", color="#cccccc", fontsize=11)
    ax.set_title(
        "ft_linear_regression — Gradient Descent Result",
        color="#ffffff",
        fontsize=13,
        fontweight="bold",
    )
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    legend = ax.legend(facecolor="#1e222a", edgecolor="#444444", labelcolor="#dddddd")

    plt.tight_layout()
    plt.savefig("plot.png", dpi=150, facecolor=fig.get_facecolor())
    print("Plot saved to plot.png")
    plt.show()


def main():
    mileages, prices, m = load_dataset()
    theta0, theta1 = load_thetas()

    if theta0 == 0.0 and theta1 == 0.0:
        print("WARNING: thetas are 0. Run 'python train.py' first.")

    print(f"theta0 = {theta0:.6f}")
    print(f"theta1 = {theta1:.6f}")
    print(f"Plotting {m} points...")
    plot_regression(mileages, prices, theta0, theta1)


if __name__ == "__main__":
    main()
