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
Correct spelling
"""
import urllib
import json
import regex as re
import logging
from collections import defaultdict
from gateway_utils import unsupported_mediatype
from flask import Flask, jsonify, request

from settings import APP_ROOT
from . import app, logger


@app.route("/intentgateway/v1/spelling/heartbeat")
def heartbeat():
    return "All good"


def words(text):
    return re.findall("[a-z0-9'.]+", text.lower())


def train(features):
    model = defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


dictionary_folder = APP_ROOT

NWORDS = train(words(file('/'.join([rulesets_folder, 'big.txt'])).read()))
alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)


def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words): return set(w for w in words if w in NWORDS)


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)


@app.route('/intentgateway/v1/spelling', methods=['POST'])
def check_spelling_post():
    if request.headers['Content-Type'].lower().startswith('text/plain'):
        return check_spelling(request.data)

    elif request.headers['Content-Type'].lower().startswith('application/json'):
        return check_spelling(json.dumps(request.json['text']))
    
    else:
        return unsupported_mediatype(request.headers['Content-Type'])


@app.route('/intentgateway/v1/spelling/<word_input>', methods=['GET'])
def check_spelling_get(word_input):
    word_input = urllib.url2pathname(word_input)
    return check_spelling(word_input)


def check_spelling(word_input):
    wordlist = words(word_input)
    correct_wordlist = []
    output = {}
    for word in wordlist:
        output_word = correct(word)
        correct_wordlist.append(output_word)
    wordsout = " ".join(correct_wordlist)
    output['words'] = wordsout
    return jsonify({'output':output})