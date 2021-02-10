from nltk import word_tokenize, sent_tokenize, bigrams
import heapq
import bs4 as bs
import urllib.request
import numpy as np
from nltk.corpus import stopwords
import re


def web_scraper(url):
    raw_html = urllib.request.urlopen(url)
    raw_html = raw_html.read()

    article_html = bs.BeautifulSoup(raw_html, features='html.parser')

    article_paragraphs = article_html.find_all('p')

    article_text = []

    for para in article_paragraphs:
        article_text.append(para.text)

    return ' '.join(article_text)


def frequency_tracker(doc):
    frequency = {}
    for sentence in doc:
        tokens = word_tokenize(sentence)
        for word in tokens:
            if word in frequency:
                frequency[word] += 1
            else:
                frequency[word] = 1
    return frequency


def document_cleaner(doc):
    corpus = sent_tokenize(doc)
    for i in range(len(corpus)):
        corpus[i] = corpus[i].lower()
        corpus[i] = re.sub(r'\W', ' ', corpus[i])
        corpus[i] = re.sub(r'\s+',' ', corpus[i])
    return corpus


def stop_word_remover(doc, stop_words):
    count = 0
    for sentence in doc:
        word_list = word_tokenize(sentence)
        doc[count] = ' '.join([token for token in word_list if not token in stop_words])  # leaves sentences as strings for other functions
        count += 1
    return doc


def bag_of_words(corpus, frequency_counts):
    sentence_vectors = [0] * len(corpus)
    outer_count = 0
    for sentence in corpus:
        tokens = word_tokenize(sentence)
        vector = [0] * len(frequency_counts)
        inner_count = 0
        for token in most_freq:
            if token in tokens:
                vector[inner_count] = 1
            inner_count += 1
        sentence_vectors[outer_count] = vector
        outer_count += 1
    return sentence_vectors


def bigram_frequency_tracker(list_tokens):
    """This function computes the frequency of all bigrams in a given list of tokens and returns key:
    value pairs of bigram: frequency in a dictionary"""
    frequency = {}
    for sentence in list_tokens:
        bg = bigrams(word_tokenize(str(sentence)))
        for bigram in bg:
            if bigram in frequency:
                frequency[bigram] += 1
            else:
                frequency[bigram] = 1
    return frequency


data = web_scraper('blank')  # replace blank with a URL 
clean_corpus = document_cleaner(data)
stop_word_list = stopwords.words('English')
clean_corpus = stop_word_remover(clean_corpus, stop_word_list)

tot_freq = frequency_tracker(clean_corpus)
most_freq = heapq.nlargest(10, tot_freq, key=tot_freq.get)

bag = bag_of_words(clean_corpus, most_freq)
BoW_model = np.asarray(bag)
print(most_freq)
print(clean_corpus[0])
print(BoW_model)
