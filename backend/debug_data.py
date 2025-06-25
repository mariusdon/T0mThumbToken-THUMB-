from generate_training_data import load_real_historical_data

print("Loading data...")
df = load_real_historical_data()

print("DataFrame info:")
print(df.info())
print("\nDataFrame head:")
print(df.head())
print("\nDataFrame columns:")
print(df.columns.tolist())
print("\nDataFrame index:")
print(df.index) 