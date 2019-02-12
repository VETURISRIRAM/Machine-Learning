import math

d1 = "The sky is blue"
d2 = "The sky is not blue"

d1Tokens = d1.split(' ')
d2Tokens = d2.split(' ')
documents = [d1Tokens, d2Tokens]
bagOfWords = list(set(list(set(d1Tokens)) + list(set(d2Tokens))))

tokenCountDict1 = dict()
for word in list(set(d1Tokens)):
	tf = d1Tokens.count(word)
	tokenCountDict1[word] = tf

tokenCountDict2 = dict()
for word in list(set(d2Tokens)):
	tf = d2Tokens.count(word)
	tokenCountDict2[word] = tf

tokenCountDict = dict()
for document in documents:
	for word in document:
		if word not in tokenCountDict:
			tokenCountDict[word] = 1
		else:
			tokenCountDict[word] += 1

idfDict = dict()
for word in bagOfWords:
	idfDict[word] = math.log((2/tokenCountDict[word]))

tfidfDict1 = dict()
for word in d1Tokens:
	tfidfDict1[word] = tokenCountDict1[word] * idfDict[word]

tfidfDict2 = dict()
for word in d2Tokens:
	tfidfDict2[word] = tokenCountDict2[word] * idfDict[word]


print("TF-IDF for document 1: ", tfidfDict1)
print("TF-IDF for document 2: ", tfidfDict2)