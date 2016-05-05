import nltk
from nltk.tokenize import word_tokenize,sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.corpus import state_union
from nltk.stem import PorterStemmer

example_test = "Hi there, Sup? Having kinda a boring day today. So whom else? Python is awesome."

#print(sent_tokenize(example_test))
#print(word_tokenize(example_test))

stop_words = set(stopwords.words("english"))

#print(stop_words)

words = word_tokenize(example_test)

filtered_words = []


# for w in words:
#     if w not in stop_words:
#         filtered_words.append(w)
#
# print(filtered_words)

# ps = PorterStemmer()
#
# new_text = "It is very important to be pythonly, while pythoning with python. All pythoners have pythoned at sometime."
#
# words = word_tokenize(new_text)
#
# for w in words:
#     print(ps.stem(w))
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            named_entity = nltk.ne_chunk(tagged)
            print(named_entity)
            # chunkGram = r"""<RB.?>*<VB.?>*<NNP>+<NN>?"""
            # chunkParser = nltk.RegexpParser(chunkGram)
            # chunked = chunkParser.parse(tagged)
            #
            # # print(tagged)
    except Exception as e:
        print(str(e))


process_content()

