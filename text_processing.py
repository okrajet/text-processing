import nltk
import string
from collections import Counter

raw_text = "Australia, officially the Commonwealth of Australia, is a sovereign country comprising the mainland of the Australian continent, the island of Tasmania, and numerous smaller islands. With an area of 7,617,930 square kilometres (2,941,300 sq mi), Australia is the largest country by area in Oceania and the world's sixth-largest country. Australia is the oldest, flattest, and driest inhabited continent, with the least fertile soils. It is a megadiverse country, and its size gives it a wide variety of landscapes and climates, with deserts in the centre, tropical rainforests in the north-east, and mountain ranges in the south-east.\nIndigenous Australians have inhabited the continent for approximately 65,000 years. The European maritime exploration of Australia commenced in the early 17th century with the arrival of Dutch explorers. In 1770, Australia's eastern half was claimed by Great Britain and initially settled through penal transportation to the colony of New South Wales from 26 January 1788, a date which became Australia's national day."

print(raw_text)

#sentence splitting

from nltk.tokenize import sent_tokenize

nltk.download('punkt')

sentences = sent_tokenize(raw_text)

print(f'There are {len(sentences)} sentences')

#tokenisation

from nltk.tokenize import word_tokenize

# word_tokenize?

tokens_list = [word_tokenize(s) for s in sentences]
#
Counter([w for x in tokens_list for w in x]).most_common(10)

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
tokens_stem = [stemmer.stem(w) for x in tokens_list for w in x]

#tokens_stem
print(Counter(tokens_stem).most_common(10))

#Exercise

from nltk.stem import SnowballStemmer, RegexpStemmer

# stemmer_s = SnowballStemmer()
# tokens_stem_s = [stemmer_s.stem(w) for x in tokens_list for w in x]

#Lemmatisation

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

nltk.download('averaged_perceptron_tagger')
tags_list = nltk.pos_tag_sents(tokens_list)
#tags_list

wordnet_tag = lambda t: 'a' if t == 'j' else (t if t in ['n', 'v', 'r'] else 'n')
lemmatizer = WordNetLemmatizer()
tokens_lemma = [lemmatizer.lemmatize(w.lower(), pos=wordnet_tag(t[0].lower())) for x in tags_list for (w, t) in x]

print(Counter(tokens_lemma).most_common(10))