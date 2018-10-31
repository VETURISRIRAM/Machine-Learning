import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Clean the dataset
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus = []
for review in df['Review']:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review = ' '.join(set(review))
    corpus.append(review)

print(corpus[0])
# Create Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(corpus).toarray()
Y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(xTrain, yTrain)

yPred = classifier.predict(xTest)

# Make Confusion Matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_true=yTest, y_pred=yPred)

# Get accuracy
from sklearn.metrics import accuracy_score
print("Accuracy : ",accuracy_score(yPred, yTest))
