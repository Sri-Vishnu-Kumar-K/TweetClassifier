from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

#print(set(synonyms))
#print(set(antonyms))

w1 = wordnet.synset("boat.n.01")
w2 = wordnet.synset("ship.n.01")

print(w1.wup_similarity(w2))