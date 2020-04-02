import nltk
from nltk.corpus import wordnet
#nltk.download('wordnet')
syns = wordnet.synsets("dog")

print(syns)