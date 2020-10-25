---
title: NLP Definitions
---
## NLP Definitions

Definitions of a few NLP related terms. I will keep updating them over time as I learn more.


### What is tokenization?

The Spacy definition:

It is the task of splitting a text (can be a word / line of text / document) into meaningful segments called tokens. The input to the tokenizer is a unicode text, and the output is a Doc object. To construct a Doc object, you need a Vocab instance, a sequence of word strings, and optionally a sequence of spaces booleans, which allow you to maintain alignment of the tokens into the original string. [^1]

### What is a vocab instance?

It is a storage class for vocabulary and other data shared across a language.

### What is lemmetization?

Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.

### What is stemming?

 Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. [^2]


### What is the difference between lemmetization & stemming?

Stemming and Lemmatization both generate the root form of the inflected words. The difference is that stem might not be an actual word whereas, lemma is an actual language word.

Stemming follows an algorithm with steps to perform on the words which makes it faster. Whereas, in lemmatization, you used WordNet corpus and a corpus for stop words as well to produce lemma which makes it slower than stemming. You also had to define a parts-of-speech to obtain the correct lemma. [^3]




[^1]: [Tokenization](https://spacy.io/usage/linguistic-features#tokenization)
[^2]: [Lemma](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
[^3]: [Difference between Stemming & Lemmetization](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)





