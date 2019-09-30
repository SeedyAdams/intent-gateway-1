"""
Routes and views for the flask application.
"""
import urllib
import json
import regex as re
import logging
from collections import defaultdict
from flask import Flask, jsonify, request

from IntentGateway import application, logger


@application.route("/heartbeat")
def heartbeat():
    return "All good"


def words(text):
    return re.findall("[a-z0-9'.]+", text.lower())


def train(features):
    model = defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


@application.errorhandler(415)
def unsupported_mediatype(type=None):
    message = {
            'status': 415,
            'message': 'Unsupported Media Type ' + type,
    }
    resp = jsonify(message)
    resp.status_code = 415

    return resp


NWORDS = train(words(file('big.txt').read()))
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


@application.route('/intentgateway/spelling', methods=['POST'])
def check_spelling_post():
    if request.headers['Content-Type'].lower().startswith('text/plain'):
        return check_spelling(request.data)

    elif request.headers['Content-Type'].lower().startswith('application/json'):
        return check_spelling(json.dumps(request.json['text']))
    
    else:
        return unsupported_mediatype(request.headers['Content-Type'])


@application.route('/intentgateway/spelling/<word_input>', methods=['GET'])
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