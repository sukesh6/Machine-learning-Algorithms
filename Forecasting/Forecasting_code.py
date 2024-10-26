# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:08:53 2024

@author: sukes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

#load the dataset by using of pandas library
solar=pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/Forecasting/Data Set (5)/solarpower_cumuldaybyday2.csv")

#convert date column to datetime column
solar['date']=pd.to_datetime(solar['date'])
solar.set_index('date', inplace = True) #setting date column as index

solar = solar.resample('D').mean()

#plot the data to visuvalise
plt.figure(figsize = (10,6))
plt.plot(solar['cum_power'], label = 'Solar Power Consumption')
plt.title('Solar Power Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Consumption')
plt.legend()
plt.show()

# Check for missing values
solar.isnull().sum()
print("Missing values:", solar.isnull().sum())
solar.fillna(method = 'ffill', inplace = True)

#plotting Trend and Seasonality using seasonal_decompose
result = seasonal_decompose(solar['cum_power'], model = 'additive', period = 365)
result.plot()
plt.show()

# Splitting the data into train and test sets (80% train, 20% test)
solar.shape
train_size = int(len(solar) * 0.8)
train_size
train = solar[:train_size]
test = solar[train_size:]
train.shape
test.shape

# 1. Exponential Smoothing (Hot-Winters)
model_exp = ExponentialSmoothing(train['cum_power'], seasonal = 'add', seasonal_periods = 365)
model_exp_fit = model_exp.fit()

#Forecasting
exp_forecast = model_exp_fit.forecast(len(test))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['cum_power'], label='Train')
plt.plot(test.index, test['cum_power'], label='Test')
plt.plot(test.index, exp_forecast, label='Exponential Smoothing Forecast')
plt.title('Exponential Smoothing Forecast')
plt.xlabel('Date')
plt.ylabel('Solar Consumption')
plt.legend()
plt.show()

#Evalute ExponentialSmoothing
mae_exp = mean_absolute_error(test['cum_power'], exp_forecast)
mae_exp
rmse_exp = np.sqrt(mean_squared_error(test['cum_power'], exp_forecast))
rmse_exp
print(f"Exponential Smoothing - MAE: {mae_exp}, RMSE: {rmse_exp}")

# 2. ARIMA Model

#plot ACF and PACF to determine AR and MA terms
plot_acf(train['cum_power'], lags = 30)
plot_pacf(train['cum_power'], lags = 30)
plt.show()

# Differencing to make the series stationary
train_dff = train['cum_power'].diff().dropna()

#Build AREMA Model
model_arima = ARIMA(train['cum_power'], order=(1,1,1))
model_arima_fit = model_arima.fit()

#Forecasting
arima_forecast = model_arima_fit.forecast(steps=len(test))

#Plotting ARIMA model
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['cum_power'], label='Train')
plt.plot(test.index, test['cum_power'], label='Test')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Solar Consumption')
plt.legend()
plt.show()

# Evaluate ARIMA Model
mae_arima = mean_absolute_error(test['cum_power'], arima_forecast)
rmse_arima = np.sqrt(mean_squared_error(test['cum_power'], arima_forecast))
print(f"ARIMA - MAE: {mae_arima}, RMSE: {rmse_arima}")

# 3. Moving Average Forecast
moving_avg = train['cum_power'].rolling(window=30).mean()

# Plotting the moving average along with actual data
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['cum_power'], label='Actual')
plt.plot(train.index, moving_avg, color='orange', label='Moving Average (30 days)')
plt.title('Moving Average - Solar Power Consumption')
plt.xlabel('Date')
plt.ylabel('Solar Consumption')
plt.legend()
plt.show()

# Model Evaluation Summary

print("\n--- Model Evaluation Summary ---")
print(f"Exponential Smoothing - MAE: {mae_exp}, RMSE: {rmse_exp}")
print(f"ARIMA - MAE: {mae_arima}, RMSE: {rmse_arima}")























































