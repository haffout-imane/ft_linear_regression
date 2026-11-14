"""Prompt for a mileage and print the estimated price using saved thetas."""

from hypothesis import estimate_price
from thetas_io import load_thetas


def read_mileage():
    """Ask until the user types a number. Mileage is km, so we expect a float."""
    while True:
        raw = input("Mileage (km): ").strip()
        if raw == "":
            print("Please enter a number.")
            continue
        try:
            return float(raw)
        except ValueError:
            print("That is not a number. Try again (example: 85000).")


def main():
    theta0, theta1 = load_thetas()
    mileage = read_mileage()
    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price}")


if __name__ == "__main__":
    main()
