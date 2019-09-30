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
Intent Gateway - intent parser
"""
import os
import errno
import urllib
import json
import nltk
import regex as re
import difflib
import itertools
import nltk.tokenize
from nltk.metrics import distance
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from flask import Flask, jsonify, request

from settings import APP_ROOT

from gateway_utils import unsupported_mediatype
from duckparser import extract_datetime
from fuzzywuzzy import fuzz
from . import app, logger

# from intent_config import intent_models
# from classifiers import vw_classifier
#from azure_router import AzureRouter, AzureRequest

stemmer = SnowballStemmer("english")
STEMMER_ENABLED = False


def literal_text_match(chat_text, rule_text, expand_candidates=True, score_threshold=0.8):
    # check for exact token overlap first
    #logger.info("literal_text_match : %s"%(rule_text))
    tokens_to_match = nltk.tokenize.word_tokenize(rule_text)
    rule_overlap = [(match_indx.span(),
                    chat_text[match_indx.start():match_indx.end()])
                    for match_indx in
                    re.finditer("\s*".join(re.escape(tok)
                                for tok in
                                tokens_to_match),
                                chat_text, re.IGNORECASE)]
    # when we have noisy text.
    # no match found so far so expanding candidates to be matched
    if len(rule_overlap) == 0:
        overlap_candidates = [chat_text[match_indx.start():
                              (match_indx.start() + len(rule_text))]
                              for match_indx in
                              re.finditer(
                                re.escape(rule_text[0:len(rule_text.split()[0])]),
                                chat_text,
                                re.IGNORECASE)]
        if expand_candidates:
            # can't find a potential candidate starting with the first word
            # greedy add character start
            if len(overlap_candidates) == 0:
                overlap_candidates = [chat_text[match_indx.start():
                                        (match_indx.start() + len(rule_text))]
                                      for match_indx in
                                      re.finditer(
                                        re.escape(rule_text[0]),
                                        chat_text,
                                        re.IGNORECASE)]
        best_candidate = ""
        if len(overlap_candidates):
            current_edit_distance = -1
            for candidate in overlap_candidates:
                d = distance.edit_distance(candidate, rule_text)
                if current_edit_distance == -1 or d <= current_edit_distance:
                    current_edit_distance = d
                    best_candidate = candidate
            if len(best_candidate):
                candidate_score = difflib.SequenceMatcher(
                                    None,
                                    best_candidate.lower(),
                                    rule_text.lower()).ratio()
                if candidate_score > score_threshold:
                    rule_overlap = [(match_indx.span(),
                                chat_text[match_indx.start():match_indx.end()])
                                for match_indx in
                                re.finditer(re.escape(best_candidate),
                                            chat_text, re.IGNORECASE)]
    return rule_overlap


def exact_match_overlap(chat_text, rule_text):
    overlap = []
    if chat_text in rule_text:
        overlap.append([(0,0),chat_text])
    return overlap


def literal_text_overlap(chat_text, rules_list, expand_candidates=True, score_threshold=0.8):
    overlap = []
    for rule_text in rules_list:
        rule_overlap = literal_text_match(chat_text, rule_text, expand_candidates, score_threshold)
        if len(rule_overlap):
            overlap.append(rule_overlap)
    return overlap


def span_overlap(chat_text, rules_list, expand_candidates=True, span_window_length=25):
    overlap = []
    for rule_text in rules_list:
        #logger.info(chat_text)
        #logger.info('hi:'+rule_text)
        # get indexes for tokens in rule_text
        tokens_to_match = nltk.tokenize.word_tokenize(rule_text)
        # tokens_to_match = rule_text.split()
        token_indexes = []

        for tok in tokens_to_match:
            single_token_overlap = literal_text_overlap(chat_text, [tok], expand_candidates=expand_candidates)
            if len(single_token_overlap):
                token_indexes.append(single_token_overlap)

        if len(token_indexes) and len(token_indexes) == len(tokens_to_match):
            # verified that all tokens in span are present in chat_text
            # going to verify that they are within span_length
            span_match = True
            current_span_index = [token_indexes[0][0][i][0][1]
                                  for i in range(len(token_indexes[0][0]))]
            for tok_index_match in token_indexes:
                tok_indx_candidates = [tok_index_match[0][i][0][1]
                                       for i in range(len(tok_index_match[0]))]
                span_diffs = [y-x for x,y in list(itertools.product(
                    current_span_index, tok_indx_candidates)) if (y-x >= 0)]
                if not len(span_diffs) or min(span_diffs) > span_window_length:
                    span_match = False
                    break
                current_span_index = tok_indx_candidates
            if span_match:
                overlap.append(token_indexes)

    return overlap


def seen_more_than(chat_text, rules_list):
    overlap = []
    for rule_tuple in rules_list:
        rule_text, count_threshold = (x.strip() for x in rule_tuple.split(';'))
        count_threshold = int(count_threshold)
        tokens_to_match = nltk.tokenize.word_tokenize(rule_text)
        rule_count = 0
        rule_overlap = []
        for tok in tokens_to_match:
            single_token_overlap = literal_text_overlap(chat_text, [tok])
            rule_count += sum(len(x) for x in single_token_overlap)
            rule_overlap.append(single_token_overlap)
        if rule_count >= count_threshold:
            overlap.append(rule_overlap)
    return overlap


def conditional_overlap(chat_text, rules_list, span_window_length=25):
    overlap = []
    for rule_text in rules_list:
        conditional_rule_overlap = span_overlap(chat_text, rule_text, expand_candidates=True)
        if len(conditional_rule_overlap) == len(rule_text):
            overlap.append(conditional_rule_overlap)
        # logger.info("conditional_overlap : %d %s %s"%(len(conditional_rule_overlap), rule_text, conditional_rule_overlap))
    return overlap
