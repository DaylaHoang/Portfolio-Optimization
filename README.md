# **Portfolio Optimization Project**

This project uses Python to fetch historical stock data, compute risk/return metrics, and optimize portfolio weights to maximize the Sharpe Ratio of a stock portfolio.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Key Metrics and Formulas](#key-metrics-and-formulas)
- [Implementation Details](#implementation-details)
  - [Data Collection](#data-collection)
  - [Log Returns](#log-returns)
  - [Covariance Matrix](#covariance-matrix)
  - [Portfolio Metrics](#portfolio-metrics)
  - [Optimization](#optimization)
- [References](#references)

---

## **Introduction**
Portfolio optimization is the methodology of choosing the proportion of different assets in order to maximize return for a certain level of risk. This project will leverage **Modern Portfolio Theory** in order to build an optimized portfolio in Python through maximization of the **Sharpe Ratio**-the measure of the risk-adjusted return of the portfolio.

---

## **Technologies Used**
- **Libraries**:
  - `yfinance`: Fetch historical stock data.
  - `pandas`: Data manipulation.
  - `numpy`: Numerical operations.
  - `scipy.optimize`: Optimization algorithms.
  - `fredapi`: Access to economic data (e.g., risk-free rate).

---

## **Project Workflow**

1. **Fetch Historical Stock Prices**: Download adjusted close prices for a list of tickers over a specified period.
2. **Compute Log Returns**: Calculate daily log returns for the portfolio.
3. **Covariance Matrix**: Measure relationships between asset returns.
4. **Portfolio Metrics**: Calculate expected return, risk (standard deviation), and the Sharpe Ratio.
5. **Optimization**: Use `scipy.optimize` to find the weights that maximize the Sharpe Ratio.

---

## **Key Metrics and Formulas**

### **1. Log Returns**
<img width="172" alt="Screenshot 2024-12-28 at 11 51 26 PM" src="https://github.com/user-attachments/assets/ec251705-1d24-4418-8dab-b309f8740581" />

### **2. Portfolio Variance**
<img width="438" alt="Screenshot 2024-12-28 at 11 52 08 PM" src="https://github.com/user-attachments/assets/7c5ab689-6292-4c8d-8a34-bb506f6330bf" />

### **3. Expected Portfolio Return**
<img width="195" alt="Screenshot 2024-12-28 at 11 52 40 PM" src="https://github.com/user-attachments/assets/c003cb95-c31e-4fff-b5fe-bb3c84f3ad2b" />

### **4. Sharpe Ratio**
<img width="481" alt="Screenshot 2024-12-28 at 11 52 50 PM" src="https://github.com/user-attachments/assets/33e8be68-1741-4186-9622-7cba800c5479" />


---

## **Implementation Details**

### **Data Collection**
- **Input**: List of tickers (e.g., `['AAPL', 'MSFT', 'GOOGL']`).
- **Source**: Yahoo Finance.
- **Output**: Adjusted close prices.

```python
tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = datetime.today() - timedelta(days=365*10)
end_date = datetime.today()

adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Adj Close']
```

---

### **Log Returns**
Logarithmic returns are computed to analyze price changes:
```python
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
```

---

### **Covariance Matrix**
The covariance matrix quantifies relationships between stock returns:
```python
cov_matrix = log_returns.cov() * 252
```

---

### **Portfolio Metrics**

#### Standard Deviation:
```python
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)
```

#### Expected Return:
```python
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252
```

#### Sharpe Ratio:
```python
def sharpe_ratio(weights, log_returns, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
```

---

### **Optimization**
Using `scipy.optimize` to find the optimal weights:
```python
from scipy.optimize import minimize

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5) for _ in range(len(tickers))]
initial_weights = np.ones(len(tickers)) / len(tickers)

optimal = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),
                   method='SLSQP', bounds=bounds, constraints=[constraints])
optimal_weights = optimal.x
```

---

## **References**
- [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Fred API Documentation](https://fred.stlouisfed.org/)
