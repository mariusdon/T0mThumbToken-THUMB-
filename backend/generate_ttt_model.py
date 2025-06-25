import os
import numpy as np
import pandas as pd
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# 1. Setup FRED
fred_api_key = "3396474600976ca3ccf351b04c2c354e"
fred = Fred(api_key=fred_api_key)

# 2. Fetch data
series = {
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "MOVE": "BAMLH0A0HYM2"
}
data = {k: fred.get_series(v, observation_start="2010-01-01") for k, v in series.items()}
df = pd.DataFrame(data).resample("M").last().dropna()

# 3. Regime
df["spread_10_2"] = df["10Y"] - df["2Y"]
df["regime"] = np.where(df["spread_10_2"] > 0.5, "steep",
                np.where(df["spread_10_2"] < -0.5, "inverted", "flat"))
regime_dummies = pd.get_dummies(df["regime"])

# 4. Bond pricing & returns
maturities = {"3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}
prices = pd.DataFrame(index=df.index)
for k, T in maturities.items():
    y = df[k] / 100
    c = y
    coupon = c * 100 / 2
    n = int(T * 2)
    prices[k] = coupon * (1 - (1 + y/2) ** -n) / (y/2) + 100 * (1 + y/2) ** -n

returns = prices.pct_change().shift(-1)
coupon_ret = df[list(maturities.keys())] / 100 / 12
total_ret = returns + coupon_ret

# drop last row
features = df[list(maturities.keys())].join(df[["CPI", "UNRATE", "MOVE"]]).join(regime_dummies).iloc[:-1]
rets = total_ret.iloc[:-1]

# 5. Targets
target_idx = rets.values.argmax(axis=1)
y = to_categorical(target_idx, num_classes=7)

# 6. Scale features
scaler = StandardScaler()
X = scaler.fit_transform(features)
joblib.dump(scaler, "scaler.pkl")

# 7. Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(7, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 9. Train
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=0)
model.save("thumb_model.h5")

# 10. Backtest
feature_dates = features.index
allocations = model.predict(X)
allocations = np.minimum(allocations, 0.4)
allocations = allocations / allocations.sum(axis=1, keepdims=True)

dates = feature_dates
ml_val = [100.0]
eq_val = [100.0]
prev_alloc = np.full(7, 1/7)

for i, date in enumerate(dates):
    ret = rets.iloc[i].values
    turnover = np.sum(np.abs(allocations[i] - prev_alloc))
    cost = turnover * 0.001
    ml_return = np.dot(allocations[i], ret)
    ml_val.append(ml_val[-1] * (1 + ml_return) * (1 - cost))
    eq_return = np.mean(ret)
    eq_val.append(eq_val[-1] * (1 + eq_return))
    prev_alloc = allocations[i]

backtest_df = pd.DataFrame({
    "date": list(dates),
    "ml_value": ml_val[1:],
    "equal_weight_value": eq_val[1:]
})
backtest_df.to_csv("backtest_results.csv", index=False) 