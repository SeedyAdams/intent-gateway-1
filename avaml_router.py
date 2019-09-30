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

from __future__ import unicode_literals
import os
from settings import APP_ROOT, MODEL_CONFIG_FILE, MODEL_ID, STOPWORDS_FILE
from . import logger, aws
import numpy as np
import joblib
from datetime import datetime
from utility.normalization import normalize_corpus
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import zipfile as zf
import spacy
nlp = spacy.load('en')
import pandas as pd


class AvaMLModel(object):
    """
    AVA ML model handler
    """
    try:
        def __init__(self, model_info):

            if type(model_info) == dict:
                self.mllib = model_info['model_lib']
                model_file = os.path.join(APP_ROOT,
                                          'configs',
                                          model_info['model_filename'])
                featex_file = os.path.join(APP_ROOT,
                                           'configs',
                                           model_info['featex_filename'])
                labels_file = os.path.join(APP_ROOT,
                                           'configs',
                                           model_info['labels_filename'])
                self.threshold = model_info['threshold']
                self.last_updated = model_info['last_updated']
            elif type(model_info) == str or type(model_info) == unicode:
                self.mllib = model_info.split('_')[0]

                # model_info
                if self.mllib == 'deep-learn-rnn-multiclass':
                    model_file = os.path.join(APP_ROOT, 'configs', '%s.json' % model_info)
                    model_weight_file = os.path.join(APP_ROOT, 'configs', '%s.h5' % model_info)
                    featex_file = os.path.join(APP_ROOT, 'configs', '%s.featex' % model_info)
                    labels_file = os.path.join(APP_ROOT, 'configs', '%s.labels' % model_info)

                elif self.mllib == 'entity':
                    entity_file = os.path.join(APP_ROOT, 'configs', '%s.zip' % model_info)

                else:
                    model_file = os.path.join(APP_ROOT, 'configs', '%s.model' % model_info)
                    featex_file = os.path.join(APP_ROOT, 'configs', '%s.featex' % model_info)
                    labels_file = os.path.join(APP_ROOT, 'configs', '%s.labels' % model_info)


                s3 = aws.active['s3']
                # logger.info("this is vzw_claims_models/%s.model"%(model_info))


                if self.mllib == 'entity':
                    # s3.download_file(Bucket='ava-intentgateway',
                    s3.download_file(Bucket=os.environ["MODEL_BUCKET_NAME"],
                                     Key="vzw_claims_models/%s.zip" % (model_info),
                                     Filename= entity_file)
                    files = zf.ZipFile(entity_file, 'r')
                    files.extractall(os.path.join(APP_ROOT, 'configs'))  # adresss to unzip
                    files.close()

                elif self.mllib == 'deep-learn-rnn-multiclass':
                    # s3.download_file(Bucket='ava-intentgateway',
                    s3.download_file(Bucket=os.environ["MODEL_BUCKET_NAME"],
                                     Key="vzw_claims_models/%s.json" % (model_info),
                                     Filename=model_file)

                    s3.download_file(Bucket=os.environ["MODEL_BUCKET_NAME"],
                                     Key="vzw_claims_models/%s.h5" % (model_info),
                                     Filename=model_weight_file)
                    # s3.download_file(Bucket='ava-intentgateway',
                    s3.download_file(Bucket=os.environ["MODEL_BUCKET_NAME"],
                                     Key="vzw_claims_models/%s.featex" % (model_info),
                                     Filename=featex_file)
                    # s3.download_file(Bucket='ava-intentgateway',
                    s3.download_file(Bucket=os.environ["MODEL_BUCKET_NAME"],
                                     Key="vzw_claims_models/%s.labels" % (model_info),
                                     Filename=labels_file)
                else:
                    # s3.download_file(Bucket='ava-intentgateway',
                    s3.download_file(Bucket=os.environ["MODEL_BUCKET_NAME"],
                                     Key="vzw_claims_models/%s.model" % (model_info),
                                     Filename=model_file)
                    # s3.download_file(Bucket='ava-intentgateway',
                    s3.download_file(Bucket=os.environ["MODEL_BUCKET_NAME"],
                                     Key="vzw_claims_models/%s.featex" % (model_info),
                                     Filename=featex_file)
                    # s3.download_file(Bucket='ava-intentgateway',
                    s3.download_file(Bucket=os.environ["MODEL_BUCKET_NAME"],
                                     Key="vzw_claims_models/%s.labels" % (model_info),
                                     Filename=labels_file)

                self.last_updated = str(datetime.now())
                self.threshold = 0.55
                self.dyna_threshold = {'Claims/deductible': 0.038,
                                       'Claims/cancel_claim': 0.023,
                                       'Claims/file_claim': 0.26,
                                       'Claims/resume_claim': 0.037,
                                       'Claims/check_claim_status': 0.17,
                                       'Claims/general_questions': 0.57,
                                       'Claims/check_claim_status/general': 0.53,
                                       'Claims/check_claim_status/tracking_info': 0.39,
                                       'Claims/check_claim_status/claim_number': 0.23,
                                       'change_claim_info':0.45,
                                       'coverage':0.45,
                                       'payment_options':0.45,
                                       'docs_affidavits': 0.45,
                                       'file_claim_problem': 0.35,
                                       'find_iphone': 0.35,
                                       'other': 0.45,
                                       'replacement_device_issue': 0.45,
                                       'screen_repair': 0.35,
                                       'issues_replacement_screen': 0.45,
                                       'screen_repair_appt_changes': 0.45,
                                       'screen_repair_coverage': 0.45,
                                       'screen_repair_issues': 0.45,
                                       'screen_repair_location': 0.45,
                                       'screen_repair_technician': 0.45,
                                       'screen_repair_info':0.45,
                                       'shipping_back_device':0.45,
                                       'need_return_label':0.45,
                                       'late_return_fees':0.45,
                                       'claims':0.45,
                                       'other_not_claims':0.45,
                                       'Claims': 0.45,
                                       'Account':0.45,
                                       'Tech_Support': 0.45,
                                       'Tech_Support/other':0.45,
                                       'Tech_Support/network/connectivity_issues':0.45
                                       }

            if self.mllib not in ['xgboost', 'scikit-learn-rf-multilabel', 'scikit-learn-rf-multiclass',
                                  'scikit-learn-lr-multiclass', 'scikit-learn-lr-multilabel','deep-learn-rnn-multiclass','entity']:
                return

            if self.mllib == 'deep-learn-rnn-multiclass':
                with open(model_file, 'r') as f:
                    self.model = model_from_json(f.read())
                self.model.load_weights(model_weight_file)
                self.featex = joblib.load(featex_file)
                self.labels = joblib.load(labels_file)

            elif self.mllib == 'entity':

                path = './IntentGateway/configs/connect_model5'
                self.model = spacy.load(path)
                self.featex = None
                self.labels = None

            else:
                self.model = joblib.load(model_file)
                self.featex = joblib.load(featex_file)
                self.labels = joblib.load(labels_file)

            logger.info('Model: %s S3 download succesful' % (model_info))

    except Exception as e:
        logger.info("appreaing here: %s" % e)

    def no_match(self, err='none'):
        _score = {}
        _score['best'] = 'No Match'
        _score['err'] = err
        return _score

    def classify_text(self, text, caller_threshold=0.55, dynamic_threshold='yes'):
        """
        classify user text
        """
        logger.info('user_text: %s' % text)
        text_entity = text.lower()

        text = normalize_corpus([''.join(text)], classifier='top_level', text_stemming=False)[0]
        logger.info('user_text(after normalization): %s' % text)

        if self.featex and not (str(type(self.featex)) == "<class 'keras_preprocessing.text.Tokenizer'>"):
            vec_feat = self.featex.transform([text])
            if not sum(vec_feat.toarray()[0]):
                _score = {}
                _labels = [(self.labels[i], 0.0) for i in range(len(self.labels))]
                _score['best'] = 'No Match'
                _score['scores'] = {}
                for _l, score in _labels:
                    _score['scores'][_l] = str(score)
                return _score

        if self.mllib == 'xgboost':
            xgbvec = xgb.DMatrix(vec_feat.toarray())
            xgb_pred = self.model.predict(xgbvec)
            xgb_score = {}
            xgb_labels = [(self.labels[i], xgb_pred[0][i]) for i in range(len(xgb_pred[0]))]
            best_pred, best_score = self.labels[np.argmax(xgb_pred[0])], xgb_pred[0][np.argmax(xgb_pred[0])]
            xgb_score['best'] = best_pred if best_score > self.threshold else 'No Match'
            xgb_score['scores'] = {}
            for _l, score in xgb_labels:
                xgb_score['scores'][_l] = str(score)
            return xgb_score


        if self.mllib in ['deep-learn-rnn-multiclass','deep-learn-rnn-multiclass_claims_v1.0.0']:

            max_length = 19
            encoded_text = self.featex.texts_to_sequences([text])
            test_vec = pad_sequences(encoded_text, maxlen=max_length, padding='post')
            _pred = self.model.predict(test_vec)

            _score = {}
            _labels = [(self.labels[i], _pred[0][i]) for i in range(len(_pred[0]))]
            best_pred, best_score = self.labels[np.argmax(_pred[0])], _pred[0][np.argmax(_pred[0])]

            if dynamic_threshold == 'yes':
                _score['best'] = best_pred if best_score > self.dyna_threshold[
                    best_pred] else 'No Match'
            else:
                _score['best'] = best_pred if best_score > float(caller_threshold) else 'No Match'
            _score['scores'] = {}
            for _l, score in _labels:
                _score['scores'][_l] = str(score)
            return _score

        elif self.mllib in ['scikit-learn-rf-multilabel', 'scikit-learn-rf-multiclass', 'scikit-learn-lr-multiclass',
                            'scikit-learn-lr-multilabel', 'scikit-learn-rf-multiclass']:

            test_vec = [vec_feat.toarray()[0]]
            _pred = self.model.predict_proba(test_vec)
            _score = {}
            _labels = [(self.labels[i], _pred[0][i]) for i in range(len(_pred[0]))]
            best_pred, best_score = self.labels[np.argmax(_pred[0])], _pred[0][np.argmax(_pred[0])]

            if dynamic_threshold == 'yes':
                _score['best'] = best_pred if best_score > self.dyna_threshold[
                    best_pred] else 'No Match'  # self.predict_with_threshold(vec_feat)
            else:
                _score['best'] = best_pred if best_score > float(caller_threshold) else 'No Match'
            _score['scores'] = {}
            for _l, score in _labels:
                _score['scores'][_l] = str(score)
            return _score

        elif self.mllib in ['entity']:
            network1 = ['data', 'hotspot', 'wifi', 'connection', 'signal', 'wireless',
                        'lte', 'network', 'wi-fi', 'service', 'cellular', 'broadband', 'tower', '3g',
                        '4g', '1x', 'booster', 'router', 'coverage', 'outage']

            text1 = text_entity.replace("?", "").replace(".","").replace(",","")
            doc2 = self.model(text1.decode('utf-8'))
            _score = {}
            _score['NETWORK'] = []
            if doc2.ents:
                for ent in doc2.ents:
                    ent_txt = ent.text
                    if ent_txt == 'wi-fi':
                        ent_txt = 'wifi'
                    if ent_txt in network1:
                        _score['NETWORK'].append(ent_txt)
            return _score

        else:
            return self.no_match('model type not supported')




class AvaMLRouter(object):
    """
    AVA ML models router
    """

    __models = None

    @classmethod
    def init_router(cls, model_info, caller_threshold="0.55", local=False):
        """
        initialize AVA ML router
        """
        try:
            r = AvaMLModel(model_info)
            return r
        except Exception, e:
            logger.error("[init_router] Can't initialize AvaML models: %s" % e)
            return e

    def __new__(cls, models, local=False):
        """
        init AVA ML handler
        """
        if AvaMLRouter.__models is None:
            AvaMLRouter.__models = object.__new__(cls)
        AvaMLRouter.__models.active = {}

        for _model, _model_info in models.iteritems():
            AvaMLRouter.__models.active[_model] = AvaMLRouter.init_router(_model_info, local)
        return AvaMLRouter.__models
