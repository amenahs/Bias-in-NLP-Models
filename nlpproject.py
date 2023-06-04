# Amenah Syed
# CS 723 Final Project
# 12/1/22

# imports & downloads

import nltk 
from nltk import pos_tag, word_tokenize
from nltk.util import ngrams 
from nltk.collocations import *
from nltk.corpus import brown, words, stopwords, wordnet as wn
from nltk.data import find
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.fasttext import FastText
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import re
import copy
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import wikipedia
warnings.filterwarnings("ignore")

en_stop = set(nltk.corpus.stopwords.words('english'))

# train & create brown model model
model_1 = Word2Vec(sentences=brown.sents(), vector_size=100, window=5, min_count=1, workers=4)
# save the brown model
model_1.save('brown.model')
brown_model = Word2Vec.load('brown.model')

# create word2vec model
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

# function to plot & display scatterplot - example code based off of https://www.kaggle.com/code/alvations/word2vec-embedding-using-gensim-and-nltk/notebook
def scatterplot(model, orig_words, word_list_to_plot):
    # plot
    labels = []
    count = 0
    max_count = len(word_list_to_plot)
    if(model == brown_model):
        X = np.zeros(shape=(max_count,len(model.wv['bias'])))
    else:
        X = np.zeros(shape=(max_count,len(model['bias'])))

    #for term in list(word2vec_model.index_to_key):
    for term in word_list_to_plot:
        if(model == brown_model):
            X[count] = model.wv[term]
        else:
            X[count] = model[term]
        labels.append(term)
        count+= 1
        if count >= max_count: break

    # recommended to use PCA first to reduce to ~50 dimensions
    pca = PCA(n_components=len(orig_words))
    X_50 = pca.fit_transform(X)

    # using TSNE to further reduce to 2 dimensions
    model_tsne = TSNE(n_components=2, random_state=0)
    Y = model_tsne.fit_transform(X_50)

    # show the scatter plot
    plt.scatter(Y[:,0], Y[:,1], 20)

    # add labels
    for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy = (x,y), xytext = (0, 0), textcoords = 'offset points', size = 10)

    # display the plot
    plt.show()
    

# find
# sample words to test
list_words = ["man", "woman", "nurse", "doctor", "Islam", "Muslim", "Christianity", "Judaism", "peace", "violence", "American",  "foreign", "Indian", "Chinese", "Japanese"]
for w in list_words:
    if w not in word2vec_model.key_to_index: # make sure all words are in the dictionary; if not, 
        list_words.delete(w)
updated_words01 = copy.deepcopy(list_words)
updated_words02 = copy.deepcopy(list_words)

# get top 10 similar words for each word in the list using Brown Corpus model
print("Most similar words using the Brown Corpus Model:")
for w in list_words:
    sim = brown_model.wv.most_similar(w, topn=10)
    list_sim01 = [s[0] for s in sim]
    list_sim01.append(w)
    print("- "+w+": ", end='')
    print(sim)
    updated_words01.extend(list_sim01)
print()

# plot
scatterplot(brown_model, list_words, updated_words01)

# get top 10 similar words for each word in the list using word2vec model
print("Most similar words using the Word2Vec Model:")
for w in list_words:
    sim = word2vec_model.most_similar(positive=[w], topn = 10)
    list_sim02 = [s[0] for s in sim]
    list_sim02.append(w)
    print("- "+w+": ", end='')
    print(sim)
    updated_words02.extend(list_sim02)
print()

# plot
scatterplot(word2vec_model, list_words, updated_words02)


# FastText webscraping - example code based off of https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/
ws1 = wikipedia.page("Muslims").content
ws1 = sent_tokenize(ws1)
# ws2 = wikipedia.page("Islam").content # this page was not recognized - see paper section 4.1.3
# ws2 = sent_tokenize(ws1)
ws3 = wikipedia.page("Nurse").content
ws3 = sent_tokenize(ws3)
ws4 = wikipedia.page("Doctor").content
ws4 = sent_tokenize(ws4)
ws5 = wikipedia.page("Man").content
ws5 = sent_tokenize(ws5)
ws6 = wikipedia.page("Woman").content
ws6 = sent_tokenize(ws6)
# ws7 = wikipedia.page("Christianity").content # this page was not recognized - see paper section 4.1.3
# ws7 = sent_tokenize(ws7)
# ws8 = wikipedia.page("Judaism").content # this page was not recognized - see paper section 4.1.3
# ws8 = sent_tokenize(ws8)
ws9 = wikipedia.page("Americans").content
ws9 = sent_tokenize(ws9)
ws10 = wikipedia.page("Indian people").content
ws10 = sent_tokenize(ws10)
ws11 = wikipedia.page("Chinese people").content
ws11 = sent_tokenize(ws11)
ws12 = wikipedia.page("Japanese people").content
ws12 = sent_tokenize(ws12)

# ws1.extend(ws2) # this page was not recognized - see paper section 4.1.3
ws1.extend(ws3)
ws1.extend(ws4)
ws1.extend(ws5)
ws1.extend(ws6)
# ws1.extend(ws7) # this page was not recognized - see paper section 4.1.3
# ws1.extend(ws8) # this page was not recognized - see paper section 4.1.3
ws1.extend(ws9)
ws1.extend(ws10)
ws1.extend(ws11)
ws1.extend(ws12)

stemmer = WordNetLemmatizer()

def preprocess_text(document):
    # remove all special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # substitute multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # convert document to lowercase
    document = document.lower()

    # lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

final_corpus = [preprocess_text(sentence) for sentence in ws1 if sentence.strip() !='']

word_punctuation_tokenizer = nltk.WordPunctTokenizer()
word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

# words representation
embedding_size = 60
window_size = 40
min_word = 5
down_sampling = 1e-2

# FastText model
ft_model = FastText(word_tokenized_corpus,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1)

semantically_similar_words = {words: [item[0] for item in ft_model.wv.most_similar([words], topn=10)]
                  for words in list_words}

# print
for k,v in semantically_similar_words.items():
    print(k+": "+str(v))
print()


# user testing
inp = input("Enter a word to find out its related words and possible bias. To quit, type QUIT (all caps) ")

while inp != "QUIT":
    # see if word is in vocabulary list. if it is, get vector representation & find 10 most similar words
    if inp in word2vec_model.key_to_index:
        print("Most similar words using the Brown Corpus Model:")
        vector = brown_model.wv[inp]  # get numpy vector of a word
        sim = brown_model.wv.most_similar(inp, topn=10)  # get other similar words
        list_sim1 = [s[0] for s in sim]
        list_sim1.append(inp)
        print(sim)
        scatterplot(brown_model, [inp], list_sim1)
        
        print("Most similar words using the Word2Vec Model:")
        sim = word2vec_model.most_similar(positive=[inp], topn = 10)
        list_sim2 = [s[0] for s in sim]
        list_sim2.append(inp)
        print(sim)
        scatterplot(word2vec_model, [inp], list_sim2)
        
    else:
        print("Sorry, that word does not exist in this dataset. Please try a different output!")
        
    print()
    inp = input("Enter a word to find out its related words and possible bias. To quit, type QUIT (all caps) ")
