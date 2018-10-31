import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:, 1:2].values
Y = df.iloc[:, 2:].values

from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
yScaler = StandardScaler()

X = xScaler.fit_transform(X)
Y = yScaler.fit_transform(Y)

from sklearn.tree import DecisionTreeRegressor
classifier = DecisionTreeRegressor(random_state=0)
classifier.fit(X, Y)

pred = classifier.predict(6.5)
print("Predicted salaray is ",pred)

xGrid = np.arange(min(X), max(X), 0.01)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.title('Decision Tree Regressor Plot')
plt.scatter(X, Y, color='blue')
plt.plot(xGrid, classifier.predict(xGrid), color='red')
plt.show()

