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

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')
MODEL_ID = 'vzw_intent_models'
MODEL_CONFIG_FILE = 'vzw_intent_models.yaml'
OSC_CONFIG = "http://localhost:5566/openscoring"
STOPWORDS_FILE = 'ava_nltk_stopwords.txt'
S3_SAGEMAKER_MODELS_BUCKET = 'ava-intentgateway'