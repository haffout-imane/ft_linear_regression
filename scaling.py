"""Min-max scale mileage to [0, 1] for GD, then convert thetas back to raw km."""


def min_max_bounds(mileages):
    """xmin and xmax of the training mileages. Span must not be 0."""
    xmin = min(mileages)
    xmax = max(mileages)
    if xmax == xmin:
        raise ValueError("All mileages are equal; cannot scale to [0, 1]")
    return xmin, xmax


def scale_to_unit_interval(mileages, xmin, xmax):
    """x' = (x - xmin) / (xmax - xmin)  →  each mileage in [0, 1]."""
    span = xmax - xmin
    return [(x - xmin) / span for x in mileages]


def denormalize_thetas(theta0_scaled, theta1_scaled, xmin, xmax):
    """
    GD ran on x'. The same line in raw x is:

        θ0' + θ1' * (x - xmin) / span
      = (θ0' - θ1' * xmin / span) + (θ1' / span) * x
    """
    span = xmax - xmin
    theta0 = theta0_scaled - theta1_scaled * xmin / span
    theta1 = theta1_scaled / span
    return theta0, theta1
