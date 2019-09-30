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
import traceback
import errno
import urllib
import json
import nltk
import regex as re
import difflib
import itertools
import trollius as asyncio
from trollius import From
from concurrent.futures import FIRST_COMPLETED
import nltk.tokenize
from nltk.metrics import distance
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from flask import Flask, jsonify, request
# from pprint import pprint

from settings import APP_ROOT

from gateway_utils import *
from duckparser import extract_datetime
from fuzzywuzzy import fuzz
from . import app, logger
from utility.normalization import normalize_corpus
#from . import azure_models
from . import avaml_models
#from . import pmml_models

from intent_config import intent_models
from avaml_router import AvaMLRouter
# from classifiers import vw_classifier
from ruleparser import literal_text_overlap, span_overlap, seen_more_than, conditional_overlap, exact_match_overlap


stemmer = SnowballStemmer("english")
STEMMER_ENABLED = False
default_caller_version = 'chat_engine_p1'
default_caller_params = {"chat_engine_env":"-", "chat_engine_model":"-"}
default_dynamic_threshold = 'no'
azure_param_model_mapper = {"Claims-dev":"azure_vzw_claims_first_dev",
            "Claims-prod":"azure_vzw_claims_first_prod",
            "Claims/check_claim_status-dev":"azure_vzw_check_claim_status_dev",
            "Claims/check_claim_status-prod":"azure_vzw_check_claim_status_prod"}
default_threshold = "0.55"


default_caller_azure_model_params = '-'.join([default_caller_params['chat_engine_model'],
                                        default_caller_params['chat_engine_env']])
default_azure_models = [azure_param_model_mapper[default_caller_azure_model_params]] if default_caller_azure_model_params in azure_param_model_mapper else []
default_model_context = {"azure":default_azure_models,
            "slot": ["dateparser_vzw_claims", "vzw_claims_info"],
            "rules": ["rulesparser_vzw_claims_first", "claims_breakout"],
            "avaml": ["rf_vzw_claims_first_v0.1.1", "gbdt_vzw_claims_first_dev","rf_multiclass_claim_status_v1.0.0", "rf_multiclass_claims_v2.0.1"]}
            # "pmml": ["pmml_osc_sample"]}


@app.route("/intentgateway/v2/intent-gateway-test")
def intent_gateway_test():
    return "all good!"


@app.route('/intentgateway/<version_number>/intent-gateway', methods=['POST'])
def classify_intent_route(version_number):
    """
    classify intent endpoint
    """
    logger.info('/intentgateway/{}/intent-gateway'.format(version_number))
    if (request.headers['Content-Type'].lower().startswith('application/json')) and (re.match('^v[1-5]$',version_number,re.IGNORECASE)):
        # logger.info(request.json)
        if 'text' not in request.json:
            logger.error('/intentgateway/{}/intent-gateway [classify_intent_route] : -text- not in request.json'.format(version_number))
            return bad_intent_gateway_request()
        try:
            chat_text = request.json['text']
            model_context = request.json['model_context'] if 'model_context' in request.json else default_model_context
            caller_version = request.json['version'] if 'version' in request.json else default_caller_version
            caller_params = request.json['params'] if 'params' in request.json else default_caller_params
            caller_threshold = request.json['threshold'] if 'threshold' in request.json else default_threshold
            caller_dynamic_threshold = request.json['dynamic_threshold'].lower() if 'dynamic_threshold' in request.json else default_dynamic_threshold
        except Exception, e:
            logger.error("/intentgateway/{}/intent-gateway [classify_intent_route] : request format error: {}".format(version_number,e))

        if type(model_context) != dict:
            return bad_model_context_request()

        try:
            ig_response = classify_intent_async(chat_text,
                                model_context,
                                caller_version,
                                caller_params,caller_threshold,caller_dynamic_threshold,
                                application_context='flask')
        except Exception, e:
            logger.error('classify_intent_route] intent classification error: {}'.fomart(e))
    else:
        return unsupported_mediatype(request.headers['Content-Type'])
    return ig_response


@app.route('/intentgateway/<version_number>/preprocess-text', methods=['POST'])
def preprocess_text(version_number):
    """
    classify intent endpoint
    """
    logger.info('/intentgateway/{}/preprocess_text'.format(version_number))
    if (request.headers['Content-Type'].lower().startswith('application/json')) and (re.match('^v[1-5]$',version_number,re.IGNORECASE)):
        # logger.info(request.json)
        if 'text' not in request.json:
            logger.error('/intentgateway/v2/preprocess-text [preprocess-text] : -text- not in request.json')
            return bad_intent_gateway_request()
        try:
            chat_text = request.json['text']
        except Exception, e:
            logger.error("/intentgateway/{}/preprocess-text [preprocess-text] : request format error: {}".format(version_number,e))

        if type(chat_text) != unicode:
            return bad_model_context_request()

        try:
            chat_text_response = normalize_corpus([''.join(chat_text)],classifier='top_level',text_stemming=False)[0]
            pre_response = {'input_text':chat_text,'preprocessed_text':chat_text_response}
        except Exception, e:
            logger.error("[classify_intent_route] intent classification error: {}".format(version_number))
    else:
        return unsupported_mediatype(request.headers['Content-Type'])
    return jsonify(pre_response)


def classify_intent(chat_text, model_context,
        caller_version=default_caller_version,
        caller_params=default_caller_params,
        application_context='local'):
    """
    classifies chat_text contextually (rules, router, trained models)
    """
    logger.info("chat_text : %s "%chat_text)
    logger.info("model_context : %s "%model_context)
    ig_response = defaultdict(dict)
    for _model_type, _models in model_context.iteritems():
        for _model in _models:
            logger.info("classifying chat_text [%s] with model %s"%(chat_text, _model))
            response = {}
            if _model_type == 'azure':
                response = azure_classify(chat_text, _model)
            elif _model_type == 'rules':
                response = rulesparser_classify(chat_text, _model)
            elif _model_type == 'slot':
                response = slot_parse(chat_text, _model)
            elif _model_type == 'avaml':
                response = avaml_classify(chat_text, _model)
            ig_response[_model_type][_model] = response
    classify_intent_response = classify_intent_summary(ig_response,
                                    caller_version,
                                    caller_params)
    if application_context == 'flask':
        return jsonify(classify_intent_response)
    else:
        return classify_intent_response


def classify_intent_async(chat_text, model_context,
        caller_version=default_caller_version,
        caller_params=default_caller_params,caller_threshold=default_threshold,dynamic_threshold=default_dynamic_threshold,
        application_context='local'):
    """
    (Async) classifies chat_text contextually (rules, router, trained models)
    """
    logger.info("chat_text : %s "%chat_text)
    logger.info("model_context : %s "%model_context)
    ig_response = defaultdict(dict)
    loop = asyncio.get_event_loop()
    tasks = []

    if caller_version == 'chat_engine_p1':
        if 'chat_engine_env' not in caller_params and 'chat_engine_model' not in caller_params:
            logger.error("[classify_intent_summary] azure model params missing (env, model)")
            return jsonify({'output':{u'error':u'request config not valid'}})
        if 'azure' not in model_context:
            return jsonify({'output':{u'error':u'request config not valid (azure not in model_context)'}})
        if caller_params['chat_engine_env'] not in ['dev', 'prod']:
            return jsonify({'output':{u'error':u'request config not valid (need dev/prod models to be specified)'}})
        if caller_params['chat_engine_model'] not in ['Claims/check_claim_status', 'Claims']:
            return jsonify({'output':{u'error':u'request config not valid (not in supported model list)'}})
        caller_azure_model_params = '-'.join([caller_params['chat_engine_model'],
                                                caller_params['chat_engine_env']])
        if caller_azure_model_params in azure_param_model_mapper:
            azure_model = azure_param_model_mapper[caller_azure_model_params]
            logger.info("classifying chat_text [%s] with model %s"%(chat_text, azure_model))
            tasks.append(asyncio.async(azure_classify(chat_text, azure_model)))
        else:
            return {}
    else:
        for _model_type, _models in model_context.iteritems():
            for _model in _models:
                logger.info("classifying chat_text [%s] with model %s"%(chat_text, _model))
                response = {}
                if _model_type == 'azure':
                    tasks.append(asyncio.async(azure_classify(chat_text, _model)))
                elif _model_type == 'rules':
                    tasks.append(asyncio.async(rulesparser_classify(chat_text, _model)))
                elif _model_type == 'slot':
                    tasks.append(asyncio.async(slot_parse(chat_text, _model)))
                elif _model_type == 'avaml':
                    tasks.append(asyncio.async(avaml_classify(chat_text, _model,caller_threshold,dynamic_threshold)))
                elif _model_type == 'entity':
                    tasks.append(asyncio.async(entity_classify(chat_text, _model)))
                elif _model_type == 'pmml':
                    tasks.append(asyncio.async(osc_classify(chat_text, _model)))

    def process_intents():
        try:
            done, pending = yield From(asyncio.wait(tasks, timeout=10))
            for fut in done:
                response = fut.result()
                if 'result' in response:
                    ig_response[response['_model_type']][response['_model']] = response['result']
            for task in pending:
                task.cancel()
        except Exception, e:
            logger.error("[classify_intent_async::process_intents] intent/entity error: %s"%e)


    loop.run_until_complete(process_intents())
    # pprint(ig_response)
    # loop.close()
    classify_intent_response = classify_intent_summary(ig_response,
                                    caller_version,
                                    caller_params)
    classify_intent_response['query'] = chat_text
    if application_context == 'flask':
        return jsonify(classify_intent_response)
    else:
        return classify_intent_response


def classify_intent_summary(ig_response,
        caller_version=default_caller_version,
        caller_params=default_caller_params):
    intents = []
    entities = []
    classify_intent_response = {}
    if caller_version == 'chat_engine_p1':
        if 'chat_engine_env' not in caller_params and 'chat_engine_model' not in caller_params:
            logger.error("[classify_intent_summary] azure model params missing (env, model)")
            return classify_intent_response
        if 'azure' not in ig_response:
            logger.error("[classify_intent_summary] azure model response missing")
            return classify_intent_response
        for _model_name, _model_results in ig_response['azure'].iteritems():
            if (caller_params['chat_engine_env'] == _model_results['env'] and
                    caller_params['chat_engine_model'] == _model_results['id']):
            # if 'dev' not in _model_name and 'prod' not in _model_name:
            #     logger.error("[classify_intent_summary] azure model must contain dev/prod in config")
            #     return classify_intent_response
                classify_intent_response = _model_results['response']
        return classify_intent_response
    for _model_type, _model_output in ig_response.iteritems():
        # pprint(_model_output)
        for _model_name, _model_results in _model_output.iteritems():
            if _model_type == 'azure':
                try:
                    _intent = {}
                    _intent['intent'] = _model_results['response']['Results']['output1'][0]['Scored Labels']
                    _intent['type'] = '%s.%s'%(_model_type, _model_name)
                    _intent['verbose'] = model_results['response']['Results']
                    intents.append(_intent)
                    # intents[_model_results['response']['Results']['output1'][0]['Scored Labels']].append((_model_type, _model_name))
                except Exception,e:
                    logger.error("[classify_intent_summary] azure model exception: %s"%e)

            elif _model_type == 'rules':
                for _intent_label in _model_results:
                    _intent = {}
                    _intent['intent'] = _intent_label
                    _intent['type'] = '%s.%s'%(_model_type, _model_name)
                    intents.append(_intent)

            elif _model_type == 'entity':
                _intent = _model_results
                _intent['type'] = '%s-%s'%(_model_type, "model")
                intents.append(_intent)

            elif _model_type == 'slot':
                for _model_subtype, _subtype_results in _model_results.iteritems():

                    if 'regex' in _model_subtype:
                        for re_tag, re_matches in _subtype_results.iteritems():
                            for re_match in re_matches:
                                entity_match = {}
                                entity_match['entity'] = re_match
                                entity_match['type'] = '%s.%s'%(_model_subtype, re_tag)
                                entity_match['id'] = '%s.%s.%s'%(_model_type, _model_name, _model_subtype)
                                entity_match['resolution'] = {"value":re_match}
                                entities.append(entity_match)

                    else:
                        for d_type, d_matches in _subtype_results.iteritems():
                            if d_type == 'time':
                                for d_match in d_matches:
                                    entity_match = {}
                                    if type(d_match['date']) == unicode:
                                        entity_match['type'] = d_match['type']
                                        entity_match['resolution'] = {"value" : d_match['date']}
                                    elif type(d_match['date']) == dict and ('to' in d_match['date']) and ('from' in d_match['date']):
                                        entity_match['type'] = d_match['type']+'range'
                                        date_range_values = []
                                        date_range_values.append({"value": d_match['date']['from'],
                                                                    "type": d_match['type'],
                                                                    "id": "from"})
                                        date_range_values.append({"value": d_match['date']['to'],
                                                                    "type": d_match['type'],
                                                                    "id": "to"})
                                        entity_match['resolution'] = {"values" : date_range_values}
                                    entity_match['entity'] = d_match['matching_entity']
                                    entity_match['id'] = '%s.%s.%s'%(_model_type, _model_name, _model_subtype)
                                    entities.append(entity_match)

            elif _model_type == 'avaml':
                try:
                    _intent = {}
                    _intent['intent'] =_model_results['best']
                    _intent['type'] = '%s.%s'%(_model_type, _model_name)
                    _intent['verbose'] = _model_results
                    # intents[_model_results['best']].append((_model_type, _model_name))
                    intents.append(_intent)
                except Exception,e:
                    logger.error("[classify_intent_summary] avaml model exception: %s"%e)

            elif _model_type == 'pmml':
                try:
                    _intent = {}
                    _intent['intent'] =_model_results['best']
                    _intent['type'] = '%s.%s'%(_model_type, _model_name)
                    _intent['verbose'] = _model_results
                    # intents[_model_results['best']].append((_model_type, _model_name))
                    intents.append(_intent)
                except Exception,e:
                    logger.error("[classify_intent_summary] pmml model exception: %s"%e)

    classify_intent_response["entities"] = entities
    classify_intent_response["intents"] = intents
    return classify_intent_response


@asyncio.coroutine
def avaml_classify(chat_text, model,caller_threshold=0.55,dynamic_threshold='no'):
    response = {}
    response['_model_type'] = 'avaml'
    response['_model'] = model
    try:
        if model not in avaml_models.active:
            avaml_models.active[model] = AvaMLRouter.init_router(model)
        result = avaml_models.active[model].classify_text(chat_text,caller_threshold,dynamic_threshold)
        logger.info("avaml_classify: [%s], classifier: %s, response: %s"%(chat_text, model, result))
        response['result'] = result
        return response
    except Exception, e:
        logger.error(traceback.print_exc())
        logger.info("avaml_classify exception: %s"%e)
        return {}

@asyncio.coroutine
def entity_classify(chat_text, model):
    response = {}
    response['_model_type'] = 'entity'
    response['_model'] = model
    try:
        if model not in avaml_models.active:
            avaml_models.active[model] = AvaMLRouter.init_router(model)

        result = avaml_models.active[model].classify_text(chat_text)
        logger.info("entity_classify: [%s], classifier: %s, response: %s" % (chat_text, model, result))
        response['result'] = result
        return response
    except Exception, e:
        logger.error(traceback.print_exc())
        logger.info("avaml_classify exception: %s" % e)
    return {}

# @asyncio.coroutine
# def osc_classify(features, model):
#     logger.info("osc_classify : %s "%features)
#     response = {}
#     response['_model_type'] = 'pmml'
#     response['_model'] = model
#     try:
#         result = pmml_models.active[model].classify_text(features, pmml_models.osc)
#         logger.info("pmml_classify: [%s], classifier: %s, response: %s"%(features, model, result))
#         response['result'] = result
#         return response
#     except Exception, e:
#         logger.info("osc_classify exception: %s"%e)
#         return {}


@asyncio.coroutine
def rulesparser_classify(chat_text, model):
    response = {}
    rulesets_to_match = {}
    chat_text = normalize_corpus([''.join(chat_text)], classifier='top_level', text_stemming=False)[0]

    if len(chat_text.split(' ')) <= 6:
        if type(model) == dict:
            response['_model'] = model.keys()[0]
            rulesets_to_match = model.values()[0]
        elif model not in intent_models['rules']:
            return {}
        else:
            response['_model'] = model
            rulesets_to_match = intent_models['rules'][model]
        response['_model_type'] = 'rules'
        logger.info('rulesets_to_match = {}'.format(rulesets_to_match.keys()))
        logger.info('rulesets_to_match values = {}'.format(rulesets_to_match.values()))

        intents = []
        fuzzyparser_match = defaultdict(list)
        if STEMMER_ENABLED:
            chat_text = ' '.join([stemmer.stem(w) for w in chat_text.split()])
        for intent_label, ruletypes in rulesets_to_match.iteritems():
            for rule_type, rules_list in ruletypes.iteritems():
                try:
                    score = fuzzy_text_overlap_score(chat_text.strip(),
                                intent_label,
                                rules_list,
                                mode=rule_type)
                    score = True if len(score) else False
                    if score:
                        fuzzyparser_match[intent_label].append(rule_type)
                except Exception as e:
                    logger.error("rulesparser_classify exception: %s "%e)

        for _label, _matched_rules in fuzzyparser_match.iteritems():
            if 'should_not_match' in _matched_rules:
                continue
            else:
                intents.append(_label)
        response['result'] = intents
    return response


@asyncio.coroutine
def slot_parse(chat_text, model):
    response = {}
    slot_parsers = {}
    if type(model) == dict:
        response['_model'] = model.keys()[0]
        slot_parsers = model.values()[0]
    elif model not in intent_models['slot']:
        return {}
    else:
        response['_model'] = model
        slot_parsers = intent_models['slot'][model]
    response['_model_type'] = 'slot'
    logger.info('slot_parsers = {}'.format(slot_parsers.keys()))
    logger.info('slot_parsers values = {}'.format(slot_parsers.values()))

    entities = defaultdict(dict)
    for _parser_id,_parser_info in slot_parsers.iteritems():
        if _parser_info['parser'] == 'duck':
            result = duck_parse(chat_text, _parser_info['type'])
            entities[_parser_id] = result
        elif _parser_info['parser'] == 'regex':
            result = regex_parse(chat_text, _parser_info['re_pattern'])
            entities[_parser_id] = result
    response['result'] = entities
    return response


def regex_parse(chat_text, re_patterns):
    # todo : optimize re.compile
    re_entity = defaultdict(list)
    for _re_pattern in re_patterns:
        re_type, re_pattern =  _re_pattern.split(';')
        re_pattern = re_pattern[1:-1]
        _re_matches = [m.group() for m in re.finditer(re_pattern, chat_text)]
        if len(_re_matches):
            re_entity[re_type] = _re_matches
    return re_entity


def duck_parse(chat_text, parse_type):
    entities = {}
    if parse_type == 'parse_time':
        date_ids = extract_datetime(chat_text)
        if len(date_ids):
            parse_time_matches = []
            for k,v in date_ids.iteritems():
                parse_time_matches.append({"matching_entity":k, "type":"builtin.datetime.date", "date":v})
            entities = {"time": parse_time_matches}
    return entities


# @asyncio.coroutine
# def azure_classify(chat_text, model):
#     logger.info("azure_classify : %s "%chat_text)
#     response = {}
#     response['_model_type'] = 'azure'
#     response['_model'] = model
#     try:
#         if model in azure_models.active:
#             result = azure_models.active[model].classify_text(chat_text)
#             response['result'] = result
#     except Exception, e:
#         logger.info("azure_classify exception: %s"%e)
#         return {}
#     logger.info("azure_classify: [%s], classifier: %s, response: %s"%(chat_text, model, response))
#     return response


# def classifier_parse(chat_text, rules_list):
#     overlap = []
#     classifier_labels = []
#     for classifier_info in rules_list:
#         classifier_type, classifier_name, classifier_version = classifier_info.split(';')
#         class_label = 'no_match'
#         if classifier_type == 'vowpal':
#             # print "going to classify with vowpal :", chat_text, classifier_name, classifier_version
#             class_label = vw_classifier.vw_classify(chat_text, classifier_name, classifier_version)
#             # print "vowpal class_label : ", class_label
#         classifier_labels.append({"type":classifier_type, "version":classifier_version, "model":classifier_name, "label":class_label})
#     if len(classifier_labels):
#         overlap.append({"intents":classifier_labels})
#     return overlap



def fuzzy_text_overlap_score(chat_text, intent_label, rules_list, mode='literal_match'):
    """returns a binary score when fuzzy rule_text is found in chat_text
        (todo: fix rule weighting after schema is decided)"""
    overlap = []
    if len(rules_list) == 0:
        return 0
    # we look for exact and fuzzy string overlap across all possible rules
    if mode == 'literal_match':
        overlap = literal_text_overlap(chat_text, rules_list)

    if mode == 'exact_match':
        overlap = exact_match_overlap(chat_text, rules_list)

    if mode == 'should_not_match':
        overlap = literal_text_overlap(chat_text, rules_list)

    if mode == 'span_match':
        overlap = span_overlap(chat_text, rules_list)

    if mode == 'seen_more_than':
        overlap = seen_more_than(chat_text, rules_list)

    if mode == 'conditional_match':
        overlap = conditional_overlap(chat_text, rules_list)

    if len(overlap):
        logger.info("FUZZYMATCH: %s, %s - %s"%(intent_label, mode, chat_text))
        # logger.info(rules_list)
        # logger.info(overlap)
    return overlap


def rule_classifier(rule_classify_request, request_type='text'):
    logger.info("rule_classifier : %s "%rule_classify_request)
    rulesets_to_match = {}
    if request_type == 'json':
        chat_text = rule_classify_request['text']
        if 'intent_context' in rule_classify_request:
            intent_context = rule_classify_request['intent_context']
        else:
            intent_context = rulesets.keys()
        for intent_group in intent_context:
            rulesets_to_match[intent_group] = rulesets[intent_group]
    else:
        return jsonify({'output':{u'error':u'input not valid'}})

    logger.info('rulesets_to_match = {}'.format(rulesets_to_match.keys()))
    logger.info('rulesets_to_match values = {}'.format(rulesets_to_match.values()))
    fuzzyparser_match = defaultdict(dict)
    # fuzzyparser_match['error'] = "raise error here"
    # if STEMMER_ENABLED:
    #     chat_text = ' '.join([stemmer.stem(w) for w in chat_text.split()])
    for intent_group, intent_labels in rulesets_to_match.iteritems():
        fuzzyparser_match[intent_group] = defaultdict(dict)
        for intent_label in intent_labels:
            fuzzyparser_match[intent_group][intent_label] = defaultdict(dict)
            for ruletype_to_match in rulesets[intent_group][intent_label]:
                try:
                    score = fuzzy_text_overlap_score(chat_text.strip(), rulesets[intent_group][intent_label][ruletype_to_match], mode=ruletype_to_match)
                    if len(score) and type(score[0]) == dict:
                        score = score
                    elif len(score):
                        score = True
                    else:
                        score = False
                except Exception as e:
                    logger.error("rule_classifier exception: %s "%e)
                    fuzzyparser_match[intent_group]['error'] = str(e)
                    score = False
                fuzzyparser_match[intent_group][intent_label][ruletype_to_match] = score
    return jsonify({'output':fuzzyparser_match})


def ruleset_cr_ranker(rules_to_rank):
    if not len(rules_to_rank) > 1:
        return
    rank_updated_idx = 0
    def swap(a, x, y):
        a[x],a[y] = a[y], a[x]
    if 'carrier specific/general/device replacement' in rules_to_rank:
        swap(rules_to_rank, rules_to_rank.index('carrier specific/general/device replacement'), rank_updated_idx)
        rank_updated_idx += 1
    if 'representative' in rules_to_rank:
        swap(rules_to_rank, rules_to_rank.index('representative'), rank_updated_idx)

