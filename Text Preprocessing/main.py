"""
@author : Sriram Veturi 
@title  : Text Preprocessing Template
"""

import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Function traverse and get the text from files
def traversal(directory):

    # To store the text of files
    contents = list()

    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            c = f.read()
            contents = contents + c.split(' ')

    return contents


# Function to preprocess the text
def preprocessText(contents, task = 1):

    # To store the corpus of words
    corpus = list()

    # Create corpus, remove special chars and lowercase operation.
    for content in contents:
        content = re.sub('[^A-Za-z0-9]+', '', content).lower()
        corpus.append(content)

    # This runs in the case where Stop Words are removed and Porter Stemmer is integrated.
    if task == 2:
        # Remove Stopwords before stemming
        corpus = [word for word in corpus if word not in stopWords]
        # Integrate Porter Stemmer
        corpus = [ps.stem(word) for word in corpus]
        # Remove Stopwords after stemming
        corpus = [word for word in corpus if word not in stopWords]

    corpus = [word for word in corpus if word != '']

    return corpus


# Function to calculate the frequency of words
def frequencyCalculate(corpus):

    # Hash table to store word frequencies.
    frequencyHash = {}

    for word in corpus:
        if word not in frequencyHash:
            frequencyHash[word] = 1
        else:
            frequencyHash[word] += 1

    # Sort the words frequencies
    sortedHash = sorted(frequencyHash.items(), key=lambda x: x[1])[::-1]
    top20 = [word for word in sortedHash]

    # Stop words in top 20 words
    stop20 = [w[0] for w in top20[:20] if w[0] in stopWords]

    return frequencyHash, top20, stop20


# Function to calculate the number of unique words accounting
# for 15% of the total words in the corpus
def uniqueAccountingFifteenPercent(top, corpus):

    threshold = 0.15 * len(corpus)

    numUniqueWords = 0
    counter = 0

    for x in top:
        if counter >= threshold:
            break
        else:
            counter += x[1]
            numUniqueWords += 1

    return numUniqueWords


# Case 1 : Without applying Stop Words removal and Integrating Porter Stemmer.
def taskOne(contents):

    corpus = preprocessText(contents)
    print("Total Number of words in the collection : \n{}\n".format(len(corpus)))
    print("Vocabulary Size : \n{}\n".format(len(set(corpus))))

    d, top, stop = frequencyCalculate(corpus)
    print("Top 20 highest frequency words : \n{}\n".format(top[:20]))
    print("Stop Words in the top 20 highest frequency words : \n{}\n".format(stop))
    print("Number of stop words in top 20 highest frequency words : \n{}\n".format(len(stop)))

    numUniqueWords = uniqueAccountingFifteenPercent(top, corpus)
    print("Number of unique words accounting 15% of total words in corpus : \n{}\n".format(numUniqueWords))


# Case 2 : After applying Stop Words removal and Integrating Porter Stemmer.
def taskTwo(contents):
    corpus = preprocessText(contents, 2)
    print("Total Number of words in the collection : \n{}\n".format(len(corpus)))
    print("Vocabulary Size : \n{}\n".format(len(set(corpus))))

    d, top, stop = frequencyCalculate(corpus)
    print("Top 20 highest frequency words : \n{}\n".format(top[:20]))
    print("Number of stop words in top 20 highest frequency words : \n{}\n".format(len(stop)))

    numUniqueWords = uniqueAccountingFifteenPercent(top, corpus)
    print("Number of unique words accounting 15% of total words in corpus : \n{}\n".format(numUniqueWords))

def initializations():

    # nltk.download()
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()
    directory = 'citeseer/citeseer'

    return stopWords, ps, directory


if __name__ == "__main__":

    # Some initializations
    stopWords, ps, directory = initializations()

    # Traverse the directory to retrieve text
    contents = traversal(directory)

    # Processing the text
    print("Case 1 : Without applying Stop Words removal and Integrating Porter Stemmer!\n")
    taskOne(contents)
    print("Case 2 : After applying Stop Words removal and Integrating Porter Stemmer!\n")
    taskTwo(contents)
