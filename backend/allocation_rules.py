import numpy as np

def compute_regime_allocation(yields):
    # yields: [3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y]
    if len(yields) != 7:
        raise ValueError("Expected 7 yields: 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y")
    yield_2y = yields[3]
    yield_10y = yields[5]
    slope = yield_10y - yield_2y
    if slope > 0.005:
        regime = "steep"
        regime_weight = {"short": 0.15, "belly": 0.20, "long": 0.65}
    elif slope < -0.005:
        regime = "inverted"
        regime_weight = {"short": 0.70, "belly": 0.20, "long": 0.10}
    else:
        regime = "flat"
        regime_weight = {"short": 0.40, "belly": 0.10, "long": 0.50}
    w_short = regime_weight["short"] / 3
    w_belly = regime_weight["belly"] / 2
    w_long_10y = regime_weight["long"] * 0.6
    w_long_30y = regime_weight["long"] * 0.4
    target = [w_short, w_short, w_short, w_belly, w_belly, w_long_10y, w_long_30y]
    target = np.array(target)
    target = np.minimum(target, 0.4)  # Cap at 40%
    target /= target.sum()  # Renormalize
    return target.tolist(), regime 