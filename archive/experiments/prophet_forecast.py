import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

file_path = r"C:\Users\adity\Downloads\MMDS\Project\Final_Processed.csv"
df = pd.read_csv(file_path)

df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
df['datetime'] = df['datetime'].dt.tz_localize(None)

df_hourly = df.groupby(pd.Grouper(key='datetime', freq='h')).agg({
    'available_cabs': 'mean'
}).reset_index()

df_hourly = df_hourly.dropna()

prophet_df = df_hourly.rename(columns={
    'datetime': 'ds',
    'available_cabs': 'y'
})

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

model.fit(prophet_df)

future = model.make_future_dataframe(periods=7 * 24, freq='h')
forecast = model.predict(future)

forecast['yhat_int'] = forecast['yhat'].round().astype(int)
forecast['yhat_lower_int'] = forecast['yhat_lower'].round().astype(int)
forecast['yhat_upper_int'] = forecast['yhat_upper'].round().astype(int)
forecast['yhat_int'] = forecast['yhat_int'].clip(lower=0)

print(forecast[['ds', 'yhat_int', 'yhat_lower_int', 'yhat_upper_int']].tail())

model.plot(forecast)
plt.title("Available Cabs Forecast")
plt.show()

model.plot_components(forecast)
plt.show()
