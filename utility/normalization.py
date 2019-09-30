# -*- coding: utf-8 -*-

###
# Intent Gateway is a natural language processing (NLP) framework to conduct model development and
# model deployment for text classification.
# Copyright (C) 2018-2019  Asurion, LLC
#
# Intent Gateway is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Intent Gateway is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Intent Gateway.  If not, see <https://www.gnu.org/licenses/>.
###

"""
Created on Fri July 18 15:45:10 2018

@author: Rakshesh
"""

import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
import unicodedata
from contractions import CONTRACTION_MAP, MAPPING_MAP
import spacy
import pandas as pd
import numpy as np
from autocorrect import spell


wnl = WordNetLemmatizer()

nlp = spacy.load('en',disabled=['parser','entity','tagger'])
tokenizer = ToktokTokenizer()
stopWord_list = pd.read_csv('./IntentGateway/utility/new_stopwords.csv')
stopWord_list = list(stopWord_list['Field'])

checkClaim_stopWord_list  = pd.read_csv('./IntentGateway/utility/check_claim_status_stopwords.csv')
checkClaim_stopWord_list = list(checkClaim_stopWord_list['Field'])

data_path = 'ig_azure_data'
no_normalize = ['claim_id','<claim_id>','phone','<phone>']
no_stem_lemma = ['started', 'filed','id','claim', 'id','number','when','num']
ava_domain_words = ['claim','number','num']

def remove_accented_chars(text):
    #text = unicode(text,encoding='utf-8')
    if type(text) != unicode:
        text = unicode(text,errors='ignore')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    new_tokens = []
    for i, t in enumerate(tokens):
        if t == '>' or (i - 1 >= 0 and tokens[i - 1]) == '<':
            continue;
        if t == '<' and (i + 1) < len(tokens) and (i + 2) < len(tokens) and (
                tokens[i + 1].lower() in no_normalize and (tokens[i + 2]) == '>'):
            joined = '<' + tokens[i + 1].lower() + '>'
            new_tokens.append(joined)
        else:
            new_tokens.append(t)
    return new_tokens


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('(%s)' % '|'.join(contraction_mapping.keys()))
    def replace(match):
        return contraction_mapping[match.group(0)]
    return contractions_pattern.sub(replace, text)


# lemmatize text based on POS tags
def lemmatize_text(text):
    if type(text) != unicode:
        text = unicode(text,errors='ignore')
    #no_lemma = ['started', 'filed']
    text = nlp(text)
    text = ' '.join(
        [word.lemma_ if (word.lemma_ != '-PRON-' and word.text.lower() not in no_stem_lemma) else word.text for word in
         text])
    return text


def stemming_text(text):
    if type(text) != unicode:
        text = unicode(text, errors='ignore')
    ps = PorterStemmer()
    text = nlp(text)
    text = ' '.join([ps.stem(word.text) if (word.text not in no_stem_lemma) else word.text for word in text])
    return text


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None,
                             [pattern.sub('', token) if token.lower() not in no_normalize else token.lower() for token
                              in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def edit_distance_mapping_replace(text,mapping_map=MAPPING_MAP):
    words = ava_domain_words
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = list()
    for token in tokens:
        mistake = token
        min_distance = float('inf')
        replace_word = mistake
        for word in words:
            ed = nltk.edit_distance(mistake, word)
            if ed < min_distance:
                min_distance = ed
                replace_word = word
        mistake = replace_word if (min_distance <= 1 and len(mistake) > 1) else mistake
        mapping_word = mapping_map.get(mistake,None)
        mapping_word = mapping_word if mapping_word is not None else mistake
        filtered_tokens.append(mapping_word)
    return ' '.join(filtered_tokens)

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1\1", text)
    text = edit_distance_mapping_replace(text)
    return ' '.join(list(map(lambda x: spell(x) if (x not in no_normalize) and (len(x) < 15) else x, text.split(' '))))


def remove_stopwords_digits(text, classifier, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    stop_list = stopWord_list
    if classifier == 'check_claim_status':
        stop_list = checkClaim_stopWord_list
    
    # remove stop words, just numeric token and token with length less than one character
    if is_lower_case:
        filtered_tokens = [token for token in tokens if
                           token not in stop_list and (not token.isdigit()) and len(token) > 1]
    else:
        filtered_tokens = [token for token in tokens if
                           token.lower() not in stop_list and (not token.isdigit()) and len(token) > 1]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, classifier='top_level',contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True,text_stemming=True,special_char_removal=True,
                     stopword_removal=True, spell_correction=True):
    normalized_corpus = []

    # normalize each document in the corpus
    for doc in corpus:
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            #print("After accented character step:",doc)
       
        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)
            #print("After contraction expansion step:",doc)
        
        #domain specific spell correction
        if spell_correction:
            doc = reduce_lengthening(doc)
            #print("After spell correction step:",doc)


        # lowercase the text
        if text_lower_case:
            doc = ' '.join([word.lower() if word.lower() not in ['id'] else word.upper() for word in doc.split(' ')])
            #print("After lower case step:",doc)
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        doc = re.sub(r'(^[0-9]{10})([^0_9])|\s+([0-9]{10})(\s+|\.+|\!+|\?+|$)|\s+([0-9]{10})(\s+|\.+|\!+|\?+)',
                     ' <phone> ', doc)
        doc = re.sub(r'(^[0-9]{7,9})([^0_9])|\s+([0-9]{7,9})(\s+|\.+|\!+|\?+|$)|\s+([0-9]{7,9})(\s+|\.+|\!+|\?+)',
                     ' <claim_id> ', doc)
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        #print("After regex replacement step:",doc)
        
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords_digits(doc, classifier,is_lower_case=text_lower_case)
            #print("After first stopwords removal step:",doc)
        
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            #print("After lemmatization step:",doc)
        # stemming text
        if text_stemming:
            doc = stemming_text(doc)
            #print("After stemming step:",doc)
        # remove special characters
        if special_char_removal:
            doc = remove_special_characters(doc)
            #print("After special character removal step:",doc)
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords_digits(doc,classifier, is_lower_case=text_lower_case)
            #print("After second stopwords removal step:",doc)
        normalized_corpus.append(doc)

    return normalized_corpus
