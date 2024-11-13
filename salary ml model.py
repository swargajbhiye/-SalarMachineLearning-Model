import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'E:\Salary_Data.csv')

x = dataset.iloc[:,:-1] #years of experience (independant variable)

y = dataset.iloc[:,-1] #salary (dependent variable)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


x_train = x_train.values.reshape(-1,1)

x_test = x_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#visulazation the test set result
plt.scatter(x_test, y_test, color ='red') # Real Salary data (testing)
plt.plot(x_train, regressor.predict(x_train), color = 'blue') # regression line from training set
plt.title('Salary vs Exprience (Test set)')
plt.xlabel('years of Experience')
plt.ylabel('Salary')
plt.show()

print(regressor)

m_slope = regressor.coef_

print(m_slope)

c_intercept = regressor.intercept_

print(c_intercept)

y_12 = m_slope * 20 + c_intercept

print(y_12)
