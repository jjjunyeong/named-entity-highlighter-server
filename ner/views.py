from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
import json
from django.contrib.auth.models import User #####
from django.http import JsonResponse , HttpResponse ####

import requests
from bs4 import BeautifulSoup

import spacy
from cleantext import clean

from joblib import Parallel, delayed
import joblib

from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras

import nltk
import numpy as np

from tensorflow_addons.layers import CRF
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word

from tqdm import tqdm

MAX_LEN = 75

def index(request):
    return HttpResponse("Hello, world. You're at the wiki index.")

def get_text_from_url(url):
    print('url:', url)
    r = requests.get(url)
    raw = BeautifulSoup(r.content, features="lxml").body.text
    return raw

def clean_text(text):
    clean_text = clean(text,
        fix_unicode=True,               # fix various unicode errors
        # to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=True,         # replace all phone numbers with a special token
        no_numbers=True,               # replace all numbers with a special token
        no_digits=True,                # replace all digits with a special token
        no_currency_symbols=True,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        # replace_with_punct="",          # instead of removing punctuations you may replace them
        # replace_with_url="<URL>",
        # replace_with_email="<EMAIL>",
        # replace_with_phone_number="<PHONE>",
        # replace_with_number="<NUMBER>",
        # replace_with_digit="0",
        # replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
    )
    
    print('text: ', clean_text[:500])
    
    return clean_text


def preprocessing(text, word2idx):
    
    sents = nltk.sent_tokenize(text)
    
    # preprocess_functions = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word]
    
    X = []
    for s in sents:
        # processed_sent = preprocess_text(s, preprocess_functions)
        words = nltk.word_tokenize(s)
        
        Xx = []
        for w in words:
            try:
                Xx.append(word2idx[w.lower()])
            except:
                Xx.append(word2idx["UNK"])
        X.append(Xx)

    # Padding each sentence to have the same lenght
    return pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])
    

def named_entity_detection_spacy(text):
    NER = spacy.load("en_core_web_sm")
    ner_text = NER(text)
    
    nes = []
    for word in ner_text.ents:
        if word.text.isalpha() and len(word.text) > 1 and len(word.text) < 30:
            nes.append(word.text)
    
    return list(set(nes))


def named_entity_detection_model(X, idx2word, idx2tag):    
    # Load NER model from file
    # NER = joblib.load('ner/model/ner_blog.h5')
    NER = keras.models.load_model('ner/model/senna_bi_lstm_crf.tf', custom_objects={'SigmoidFocalCrossEntropy': SigmoidFocalCrossEntropy()})
    
    nes = []
    
    for i in tqdm(range(X.shape[0])):
        p = NER.predict(np.array([X[i]]), verbose=None)
        p = np.argmax(p, axis=-1)
        
        for w, t in zip(X[i], p[0]):
            # if word is not 'PAD' or 'UNK' and tag is not 'PAD' or 'O'
            if w != 0 and w != 1 and t != 0 and t != 1 and len(idx2word[w])>2: 
                nes.append(idx2word[w])
    
    return list(set(nes))


def get_named_entities(request):
    print('get raw text...')
    url = request.GET.get('url', None)
    raw = get_text_from_url(url)
    
    print('preprocessing...')
    
    idx2word = joblib.load('ner/embedding/senna_idx2word.pkl')
    idx2tag = joblib.load('ner/embedding/senna_idx2tag.pkl')
    
    word2idx = {w: i for i, w in idx2word.items()}
    
    text = clean_text(raw)
    X = preprocessing(text, word2idx)
    print('X shape: ', X.shape)
    
    # nes = named_entity_detection(text)
    print('model prediction...')
    nes = named_entity_detection_model(X, idx2word, idx2tag)

    data = {
        'named_entity': "/".join(nes),
        'alert': 'success',
    }

    print('json-data to be sent: ', data)

    return JsonResponse(data)
