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
# print (Counter([w for x in tokens_list for w in x]).most_common(10))
print (Counter([w for x in tokens_list for w in x]))

