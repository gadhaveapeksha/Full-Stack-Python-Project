import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\gadha\Downloads\Salary_Data.csv')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

print(regressor)
print(regressor.get_params())

y_pred = regressor.predict(x_test)
print(y_pred)

comparision = pd.DataFrame({'Actual':y_test,'Prediction': y_pred})
print(comparision)

plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary of employee based on experiance')
plt.xlable('Experiance')
plt.ylable('Salary')
plt.show()