"""Batch gradient descent pieces from the subject: cost J, tmp thetas, one update."""

from scaling import denormalize_thetas, min_max_bounds, scale_to_unit_interval


def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


def cost_j(mileages, prices, theta0, theta1):
    """
    J(theta0, theta1) = (1 / (2m)) * sum (estimate - price)^2

    """
    m = len(mileages)
    total = 0.0
    for i in range(m):
        error = estimate_price(mileages[i], theta0, theta1) - prices[i]
        total += error * error
    return total / (2 * m)


def compute_tmps(mileages, prices, theta0, theta1, learning_rate):
    """
    Subject formulas:

    tmpθ0 = alpha * (1/m) * Σ (estimatePrice(mileage[i]) - price[i])
    tmpθ1 = alpha * (1/m) * Σ (estimatePrice(mileage[i]) - price[i]) * mileage[i]

    Both sums use the SAME current theta0 and theta1 (simultaneous).
    """
    m = len(mileages)
    sum_error = 0.0
    sum_error_x = 0.0
    for i in range(m):
        error = estimate_price(mileages[i], theta0, theta1) - prices[i]
        sum_error += error
        sum_error_x += error * mileages[i]
    tmp0 = learning_rate * (1 / m) * sum_error
    tmp1 = learning_rate * (1 / m) * sum_error_x
    return tmp0, tmp1


def gradient_step(mileages, prices, theta0, theta1, learning_rate):
    tmp0, tmp1 = compute_tmps(mileages, prices, theta0, theta1, learning_rate)
    return theta0 - tmp0, theta1 - tmp1


def fit_linear_regression(
    mileages,
    prices,
    learning_rate=0.1,
    n_iterations=10000,
    log_every=1000,
):
    xmin, xmax = min_max_bounds(mileages)
    scaled = scale_to_unit_interval(mileages, xmin, xmax)

    theta0_s = 0.0
    theta1_s = 0.0
    print(f"J start (scaled x): {cost_j(scaled, prices, theta0_s, theta1_s)}")

    for iteration in range(1, n_iterations + 1):
        theta0_s, theta1_s = gradient_step(
            scaled, prices, theta0_s, theta1_s, learning_rate
        )
        if log_every and iteration % log_every == 0:
            print(f"J iter {iteration}: {cost_j(scaled, prices, theta0_s, theta1_s)}")

    theta0, theta1 = denormalize_thetas(theta0_s, theta1_s, xmin, xmax)
    print(f"J end (raw x, denormalized theta): {cost_j(mileages, prices, theta0, theta1)}")
    return theta0, theta1


if __name__ == "__main__":
    # --- Toy from Part A: (x,y) = (1,3), (2,5); start at 0; alpha = 0.1 ---
    xs = [1.0, 2.0]
    ys = [3.0, 5.0]
    alpha = 0.1
    print("=== toy dataset, one step ===")
    print("J before:", cost_j(xs, ys, 0.0, 0.0))
    tmp0, tmp1 = compute_tmps(xs, ys, 0.0, 0.0, alpha)
    print("tmp0 (expect -0.4):", tmp0)
    print("tmp1 (expect -0.65):", tmp1)
    t0, t1 = gradient_step(xs, ys, 0.0, 0.0, alpha)
    print("new theta0 (expect 0.4):", t0)
    print("new theta1 (expect 0.65):", t1)
    print("J after: ", cost_j(xs, ys, t0, t1))

    # --- Real CSV, still ONE step, no scaling: θ1 explodes ---
    from dataset import load_dataset

    mileages, prices, m = load_dataset()
    print("\n=== data.csv, one step, alpha=0.1, NO scaling ===")
    print("m =", m)
    print("J before:", cost_j(mileages, prices, 0.0, 0.0))
    raw0, raw1 = gradient_step(mileages, prices, 0.0, 0.0, 0.1)
    print("new theta0:", raw0)
    print("new theta1:", raw1)
    print("J after: ", cost_j(mileages, prices, raw0, raw1))
    print("If J went up and theta1 is huge, that is why we scale mileage.")

    print("\n=== data.csv, 10000 steps, mileage scaled to [0, 1] ===")
    theta0, theta1 = fit_linear_regression(mileages, prices)
    print("theta0 (raw):", theta0)
    print("theta1 (raw):", theta1)
    print("estimate at 50000 km:", estimate_price(50000, theta0, theta1))
    print("estimate at 200000 km:", estimate_price(200000, theta0, theta1))
    print("theta1 should be negative; 200000 km should be cheaper than 50000.")
