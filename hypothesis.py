"""Hypothesis from the subject: estimatePrice(mileage) = theta0 + theta1 * mileage."""


def estimate_price(mileage, theta0, theta1):
    """Return the estimated price for one mileage, given current parameters."""
    return theta0 + theta1 * mileage


if __name__ == "__main__":
    # Untrained model: both thetas are 0, so every mileage maps to price 0.
    print("untrained, mileage 100000:", estimate_price(100000, 0, 0))

    # Toy line h(x) = 1 + 2x  (from Part A)
    print("toy h(3) with theta0=1, theta1=2:", estimate_price(3, 1, 2))
