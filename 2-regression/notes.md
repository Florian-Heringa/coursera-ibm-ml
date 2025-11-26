# Supervised Machine Learning - Regression

## Error measures
- **Sum of Squared Error** => Squared error between target variable and prediction
- **Total Sum of Squares** => Variance of original data
- **Coefficient of Determination ($R^2$)** => How well can the model explain the variation in the data

> $R^2 = 1 - \frac{SSE}{TSS}$

## Using sklearn for regression

A quick example for how to fit a linear regression model:

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr = lr.fit(X_train, y_train)
y_predict = lr.predict(X_Text)
```