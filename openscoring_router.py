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
import json
import openscoring

from settings import APP_ROOT, OSC_CONFIG
from . import app, logger
from intent_config import intent_models

# OSC_CONFIG = "http://localhost:5566/openscoring"

class OpenscoringModel(object):
    """
    Openscoring model handler
    """

    def __init__(self, model_info, osc):
        self.kwargs = {"auth" : ("admin", "adminadmin")}
        self.mllib = model_info['model_lib']
        if self.mllib in ['random_forest_pmml', 'svm_pmml', 'tensorflow_pmml']:
            osc.deploy(model_info['model_id'],
                        os.path.join(APP_ROOT, 'configs',model_info['model_filename']),
                        **self.kwargs)
        self.model_id = model_info['model_id']
        self.last_updated = model_info['last_updated']
        self.threshold = model_info['threshold']

    def no_match(self, err='none'):
        _score = {}
        _score['best'] = 'No Match'
        _score['err'] = err
        return _score

    def classify_text(self, features, osc):
        """
        classify user text
        """
        _score = {}
        if not type(features) == dict:
            return self.no_match('features not valid')
        if self.mllib == 'random_forest_pmml':
            _score['best'] = 'OSC Integration Demo'
            _score['result'] = osc.evaluate(self.model_id, features)
            return _score
        else:
            return self.no_match('model type not supported')

class OpenscoringRouter(object):
    """
    Openscoring - pmml model router
    """

    __models = None

    @classmethod    
    def init_router(cls, model_info, local=False):
        """
        initialize Openscoring router
        """
        try:
            return OpenscoringModel(model_info, OpenscoringRouter.__models.osc)
        except Exception, e:
            logger.error("[init_router] Can't initialize Openscoring models: %s"%e)
            return e

    def __new__(cls, models, local=False):
        """
        init AWS session handler
        """
        if OpenscoringRouter.__models is None:
            OpenscoringRouter.__models = object.__new__(cls)
        OpenscoringRouter.__models.active = {}
        OpenscoringRouter.__models.osc = openscoring.Openscoring(OSC_CONFIG)
        
        for _model, _model_info in models.iteritems():
            OpenscoringRouter.__models.active[_model] = OpenscoringRouter.init_router(_model_info, local)
        return OpenscoringRouter.__models
