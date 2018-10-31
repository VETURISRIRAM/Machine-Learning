import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Modelling.csv')

X = df.iloc[:, 3 : 13].values
Y = df.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
xScaler = StandardScaler()
yScaler = StandardScaler()

xTrain = xScaler.fit_transform(xTrain)
xTest = xScaler.transform(xTest)

# Now, let's make the ANN

# Import Keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN
classifier = Sequential()

# Add Input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# Add the next hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Add the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Make the predictions
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to training set
classifier.fit(xTrain, yTrain, batch_size=10, nb_epoch=100)

yPred = classifier.predict(xTest)
yPred = (yPred > 0.5)
# Confusion Matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(yPred, yTest)

# Accuracy on test set
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(yTest, yPred)
print("Accuracy on test set :", accuracy)
