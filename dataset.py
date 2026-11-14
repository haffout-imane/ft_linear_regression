"""Load the mileage/price CSV used to train the linear regression."""

import csv
from pathlib import Path


DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "data.csv"


def load_dataset(csv_path=DEFAULT_CSV):
    """
    Read a CSV with a header `km,price` and one example per following row.

    Returns:
        mileages: list of float  (x in the subject: mileage[i])
        prices:   list of float  (y in the subject: price[i])
        m:        int            (number of training examples)

    mileages[i] and prices[i] are the same car. m == len(mileages) == len(prices).
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    mileages = []
    prices = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or unreadable CSV: {path}")

        required = {"km", "price"}
        got = {name.strip() for name in reader.fieldnames}
        if not required.issubset(got):
            raise ValueError(
                f"CSV must have columns km and price, got: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            km_raw = (row.get("km") or "").strip()
            price_raw = (row.get("price") or "").strip()
            if km_raw == "" and price_raw == "":
                continue
            try:
                mileage = float(km_raw)
                price = float(price_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric data at line {row_number}: {row}"
                ) from exc
            mileages.append(mileage)
            prices.append(price)

    m = len(mileages)
    if m == 0:
        raise ValueError(f"No training examples in {path} (m would be 0)")

    return mileages, prices, m


if __name__ == "__main__":
    xs, ys, m = load_dataset()
    print(f"m = {m}")
    print(f"first example: km={xs[0]}, price={ys[0]}")
    print(f"last example:  km={xs[-1]}, price={ys[-1]}")
    print(f"km range:    {min(xs)} .. {max(xs)}")
    print(f"price range: {min(ys)} .. {max(ys)}")
