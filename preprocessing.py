
import nltk
import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
# FILE location can be replaced as necessary
DATA_FILE = 'recovery-news-data.csv'
STOPWORDS_FILE = 'stopwords.txt'
STOPWORDS = {}
PREPROCESS_OUTPUT = 'preprocessed-recovery-news-data.csv'

with open(STOPWORDS_FILE) as f:
    STOPWORDS = set(f.read().splitlines())
ACRONYMS = {
    'W.H.O.': 'WHO',
    'U.S.': 'US',
    'C.D.C.': 'CDC',
    'PH.D': 'PHD',
    'DR.': 'DR',
    'MR.': 'MR,',
    'MRS.': 'MRS',
    'MS.': 'MS'
}

WORD_BLACKLIST = {
    '...'
}

nltk.download('omw-1.4')
nltk.download('wordnet')
lemma = nltk.wordnet.WordNetLemmatizer()


def preProcessArticle(txt):
    # parse stopwords file
    # remove links
    txt = re.sub('http\S+', '', txt)
    txt = re.sub('www\S+', '', txt)
    # remove dashes
    txt = txt.replace('-', ' ')
    # remove unicode
    txt = re.sub('[^\.a-zA-Z0-9\s ]', ' ', txt)

    txt = txt.upper()
    for key in ACRONYMS:
        while key in txt:
            txt = txt.replace(key, ACRONYMS[key])
    txt = txt.lower()
    # split text into sentences and then words
    sentencelist = re.split('[.\n]', txt)
    for i, sentence in enumerate(sentencelist):
        wordlist = sentence.split()
        for j, word in enumerate(wordlist):
            if word.lower() in STOPWORDS or word.lower() in WORD_BLACKLIST:
                wordlist[j] = ''
            else:
                wordlist[j] = lemma.lemmatize(word)
        sentencelist[i] = ' '.join(wordlist)
    txt = '. '.join([s for s in sentencelist if len(s) > 0])

    # standardize
    txt = txt.replace('\n', ' ')
    while ('  ' in txt):
        txt = txt.replace('  ', ' ')
    txt = txt.lower()
    return txt.strip()


data = pd.read_csv(DATA_FILE)
data['title'] = data['title'] + '.'
data['combined'] = data['title'] + data['body_text']
data['body_text_processed'] = data['combined'].astype(str).apply(preProcessArticle)

data.to_csv(PREPROCESS_OUTPUT)
