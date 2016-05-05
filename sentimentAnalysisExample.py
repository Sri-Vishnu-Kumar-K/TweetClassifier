import nltk
import random
from nltk.corpus import movie_reviews
import pickle
document = []

for categories in movie_reviews.categories():
    for fileid in movie_reviews.fileids():
        document.append((list(movie_reviews.words(fileid)),categories))

random.shuffle(document)

#print(document[1])

allWords = []

for w in movie_reviews.words():
    allWords.append(w.lower())

allWords = nltk.FreqDist(allWords)

chosenWords = list(allWords.keys())[:3000]

def find_features(docu):
    words = set(docu)
    features = {}
    for w in chosenWords:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev),category) for (rev,category) in document]

#print(featuresets[0])

train_set = featuresets[:1900]
test_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print("The accuracy is :", nltk.classify.accuracy(classifier,test_set)*100)

classifier.show_most_informative_features(15)

saveclassifier = open("G:\PythonProjects\NaiveBayes.pickle","wb")
pickle.dump(classifier,saveclassifier)
saveclassifier.close()

