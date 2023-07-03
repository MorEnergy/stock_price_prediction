# Stock Price Prediction

This project focuses on using historical stock market data to build predictive models for forecasting future stock prices. The goal is to evaluate the performance of different machine learning algorithms and gain insights into stock market trends.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description
In this project, we fetch historical stock data using the `yfinance` library and perform feature engineering to extract relevant information. We calculate additional features such as daily returns, rolling volatility, and more. Then, we split the data into training and testing sets, and select machine learning models for prediction.

The models used in this project include:
- Linear Regression
- Random Forest
- Hist Gradient Boosting

We use hyperparameter tuning with GridSearchCV to find the best model for the given dataset. Finally, we evaluate and compare the models based on root mean squared error (RMSE) and R-squared values.

## Installation
1. Clone this repository to your local machine.
2. Install the required libraries by running `pip install -r requirements.txt` in your terminal.

## Usage
1. Set the stock ticker symbol, start date, and end date in the code.
2. Run the script to fetch the historical stock data and perform prediction.
3. View the evaluation metrics and forecasted prices.

## Contributing
Contributions are welcome! If you have any suggestions, enhancements, or bug fixes, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

