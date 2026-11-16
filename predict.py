"""Prompt for a mileage and print the estimated price using saved thetas."""

from dataset import load_dataset
from gradient_descent import estimate_price
from thetas_io import load_thetas


def read_mileage():
    while True:
        raw = input("Mileage (km): ").strip()
        if raw == "":
            print("Please enter a number.")
            continue
        try:
            return float(raw)
        except ValueError:
            print("That is not a number. Try again (example: 85000).")


def warn_if_unreliable(mileage, price):
    if mileage < 0:
        print("Warning: negative mileage is not physically valid.")

    try:
        mileages, _, _ = load_dataset()
    except (FileNotFoundError, ValueError):
        mileages = []

    if mileages and not min(mileages) <= mileage <= max(mileages):
        print(
            f"Warning: the entered mileage is outside the training range "
            f"({min(mileages):g}–{max(mileages):g} km). "
            "This is extrapolation, so the estimate may be unreliable."
        )

    if price < 0:
        print(
            "Warning: the linear model can produce negative prices; "
            "this estimate is not realistic."
        )


def main():
    theta0, theta1 = load_thetas()
    mileage = read_mileage()
    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price}")
    warn_if_unreliable(mileage, price)
if __name__ == "__main__":
    main()
