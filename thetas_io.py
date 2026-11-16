"""Save and load theta0 and theta1. Missing file means untrained: both are 0."""

import json
from pathlib import Path


DEFAULT_THETAS = Path(__file__).resolve().parent / "thetas.json"


def save_thetas(theta0, theta1, path=DEFAULT_THETAS):
    path = Path(path)
    payload = {"theta0": float(theta0), "theta1": float(theta1)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_thetas(path=DEFAULT_THETAS):
    path = Path(path)
    if not path.is_file():
        return 0.0, 0.0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["theta0"]), float(data["theta1"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid theta file {path}: {exc}") from exc


if __name__ == "__main__":
    t0, t1 = load_thetas()
    print(f"load default path (0, 0 if untrained): theta0={t0}, theta1={t1}")

    demo = Path(__file__).resolve().parent / "_demo_thetas.json"
    save_thetas(1.5, -0.02, demo)
    loaded = load_thetas(demo)
    print(f"saved then loaded demo file: {loaded}")
    demo.unlink()
    print("removed demo file")
