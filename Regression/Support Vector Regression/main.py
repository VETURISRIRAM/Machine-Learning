import pandas as pd

df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:, 1:2].values
Y = df.iloc[:, 2:].values

from sklearn.preprocessing import  StandardScaler
xScaler = StandardScaler()
yScaler = StandardScaler()

X = xScaler.fit_transform(X)
Y = yScaler.fit_transform(Y)

from sklearn.svm import SVR
classifier = SVR(kernel='rbf')
classifier.fit(X, Y)

pred = classifier.predict(6.5)
print("Prediction for X=6.5 is : ",pred)

import matplotlib.pyplot as plt
plt.title('SVR Rgeression')
plt.scatter(X, Y, color='blue')
plt.plot(X, classifier.predict(X), color='red')
plt.show()