import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Fetch historical stock data
ticker = 'JLR'
start_date = '2010-01-01'
end_date = '2021-12-31'

data = yf.download(ticker, start=start_date, end=end_date)
data = data.dropna()

# Step 2: Feature engineering
data['Date'] = data.index
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Calculate additional features
data['DailyReturn'] = data['Close'].pct_change()
data['LogReturn'] = np.log(1 + data['DailyReturn'])
data['RollingVolatility'] = data['LogReturn'].rolling(window=30).std() * np.sqrt(252)

# Step 3: Define features and target variable
features = ['Year', 'Month', 'Day', 'RollingVolatility']
target = 'Close'

X = data[features]
y = data[target]

# Step 4: Handle missing values
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model selection and hyperparameter tuning
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Hist Gradient Boosting': HistGradientBoostingRegressor()
}

param_grids = {
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10]},
    'Hist Gradient Boosting': {'learning_rate': [0.01, 0.1, 0.2], 'max_iter': [100, 200, 300]}
}

best_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    best_models[model_name] = model

# Step 7: Evaluate and compare the models
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, R-squared: {r2:.2f}")


# Step 8: Forecast future stock prices using the best model
future_dates = pd.date_range(start=end_date, periods=30, freq='D')

# Merge future data with original data
future_data = pd.DataFrame({'Date': future_dates})
data = data.reset_index(drop=True)
future_data = future_data.merge(data, how='left', on='Date')

# Calculate additional features for future data
future_data['RollingVolatility'] = future_data['LogReturn'].rolling(window=30).std() * np.sqrt(252)

# Select features for future predictions
future_data = future_data[features]

# Step 9: Make predictions on future data
best_model = best_models['Hist Gradient Boosting']
future_predictions = best_model.predict(future_data)

# Step 10: Display the forecasted prices
forecast = pd.DataFrame({'Date': future_dates, 'Forecast': future_predictions})
print(forecast)

