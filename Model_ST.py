import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error


df = pd.read_csv('US_CARRIER_ONLY.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')


all_forecasts = []

# Filter airlines with consistent data in the past 6 months
recent_date = df['Date'].max()
six_months_ago = recent_date - pd.DateOffset(months=6)
airlines_with_recent_data = df[(df['Date'] >= six_months_ago)].groupby('Airline Name').filter(lambda x: x['Date'].nunique() == 6)['Airline Name'].unique()


for airline in airlines_with_recent_data:
    
    airline_data = df[df['Airline Name'] == airline]
    airline_data = airline_data.set_index('Date').sort_index()

    
    train_data = airline_data[airline_data.index.year < 2024]
    test_data = airline_data[airline_data.index.year == 2024]

    
    if len(test_data) == 0:
        continue

    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data['Flights_Scaled'] = scaler.fit_transform(train_data[['Flights']])

   
    ets_model = ExponentialSmoothing(train_data['Flights_Scaled'], seasonal='add', seasonal_periods=12, trend='add')
    ets_fit = ets_model.fit()
    ets_forecast_train = ets_fit.forecast(steps=len(test_data))

    
    train_data['Residuals'] = train_data['Flights_Scaled'] - ets_fit.fittedvalues

    
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length - 1):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    
    seq_length = 5
    X_residuals, y_residuals = create_sequences(train_data['Residuals'].dropna().values, seq_length)

    
    xgb_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_residuals, y_residuals)

    
    X_test_residuals, _ = create_sequences(train_data['Residuals'].dropna().values, seq_length)
    xgb_residual_forecast_test = xgb_model.predict(X_test_residuals[-len(test_data):])

    
    final_scaled = np.array(ets_forecast_train).reshape(-1, 1) + np.array(xgb_residual_forecast_test).reshape(-1, 1)
    final = scaler.inverse_transform(final_scaled).flatten()

    #MAPE
    mape = mean_absolute_percentage_error(test_data['Flights'], final)
    print(f"MAPE for {airline}: {mape:.2%}")

    
    forecast_dates = pd.date_range(start=train_data.index[-1] + pd.DateOffset(months=1), periods=len(test_data), freq='M')
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Airline': airline,
        'Predicted_Flights': final
    })
    
    all_forecasts.append(forecast_df)


final_forecast_df = pd.concat(all_forecasts).reset_index(drop=True)
print(final_forecast_df)
