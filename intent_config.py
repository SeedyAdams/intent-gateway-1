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

import os
import sys
import yaml
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer

from settings import APP_ROOT, MODEL_CONFIG_FILE, MODEL_ID, STOPWORDS_FILE
# from . import app, logger

# stemmer = SnowballStemmer("english")
# STEMMER_ENABLED = False

# stopwords = defaultdict(int)
# for line in open(os.path.join(APP_ROOT,
#                 'configs',
#                 STOPWORDS_FILE),'r'):
#     stopwords[line.strip()] += 1

# print("stopwords file : {}, num stopwords : {}".format(stopwords_file, len(stopwords)))
# print('has' in stopwords)

intent_models = defaultdict(dict)
# logger.info("loading configs for intent classification and entity extraction models...")

def load_configs():
    """
    load intent model configs
    """
    config_file = os.path.join(APP_ROOT, 'configs', MODEL_CONFIG_FILE)
    intent_model_configs = yaml.load(open(config_file, 'r').read())
    if MODEL_ID not in intent_model_configs:
        raise Exception("invalid model config : missing MODEL_ID {0}".format(MODEL_ID))
    for model, model_info in intent_model_configs[MODEL_ID].iteritems():
        # print model, model_info
        for _config_file in model_info['config_files']:
            try:
                _model_config = yaml.load(open(os.path.join(APP_ROOT, 'configs', _config_file)))
                intent_models[model_info['model_type']].update(_model_config)
            except Exception,e:
                print "Exception : %s"%e

# load_configs()

# logger.info("Done loading configs for intent models : %s"%(",".join(intent_models.keys())))
_model_info = ''
for model_type, models in intent_models.iteritems():
    _model_info += "%s\t[%s]\n"%(model_type, ", ".join(models.keys()))
print "Done loading configs for intent models : %s"%(_model_info)
