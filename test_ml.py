import pandas as pd
from ml_predictor import MLPredictor

# Load historical data
df = pd.read_csv('historical_data/correlation_data.csv', parse_dates=['date'])
df = df.set_index('date')
print(f"Loaded {len(df)} rows of historical data")
print(f"Columns: {df.columns.tolist()}")

predictor = MLPredictor()
print(f"Predictor loaded: {predictor.loaded}")

iv_data = {'index_iv': 20, 'vix_level': 18}
z_score = 0.31
impl_corr = 0.277

try:
    result = predictor.predict(iv_data, z_score, impl_corr, df)
    print('Result:', result)
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
