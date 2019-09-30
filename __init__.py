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
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask
from flask_cors import CORS


from app_logging import Logger
logfilename =  os.environ["LOG_FILENAME"] if "LOG_FILENAME" in os.environ and len(os.environ["LOG_FILENAME"]) else '/var/log/intent-gateway.log'
logger = Logger(filename=logfilename)
logger.info("Intent Gateway #-> initializing...")

LOCAL_CONTAINER = True if "RUN_LOCAL_CONTAINER" in os.environ and os.environ["RUN_LOCAL_CONTAINER"]=='1' else False

if LOCAL_CONTAINER == 1:
    app = Flask(__name__, static_folder='./static')
    print("RUNNING THIS LOCALLY")
    logger.info('Running this LOCALLY')
else:
    app = Flask(__name__, static_folder=None)
    app = Flask(__name__, static_url_path='/intentgateway/static')
    print("RUNNING THIS REMOTE")
    logger.info('Running this REMOTE')

CORS(app, resources={r"/": {"origins": "*"}})



from aws_clients import AWSClients
aws = AWSClients(['s3', 'kinesis'], local=LOCAL_CONTAINER)
logger.info("active AWS clients : %s"%aws.active)

import intent_config
intent_config.load_configs()
#logger.info("intent_config.intent_models : %s"%intent_config.intent_models)

#import api_demo
#import duckparser

##test comment

# default parsers, models loaded at init time
# skip_duckparser =  os.environ["SKIP_DUCKPARSER"] if "SKIP_DUCKPARSER" in os.environ and len(os.environ["SKIP_DUCKPARSER"]) else False
# if not skip_duckparser:
#     duckparser.init_duck()
    

# azure_models = None
# from azure_router import AzureRouter
# if 'azure' in intent_config.intent_models:
#     azure_models = AzureRouter(intent_config.intent_models['azure'])


# pmml_models = None
# from openscoring_router import OpenscoringRouter
# if 'pmml' in intent_config.intent_models:
#     logger.info("pmml -> %s"%intent_config.intent_models['pmml'])
#     pmml_models = OpenscoringRouter(intent_config.intent_models['pmml'])
# logger.info("pmml_models.active : %s"%pmml_models.active)

avaml_models = None
from avaml_router import AvaMLRouter
if 'avaml' in intent_config.intent_models:
    logger.info("avaml -> %s"%intent_config.intent_models['avaml'])
    avaml_models = AvaMLRouter(intent_config.intent_models['avaml'])
logger.info("avaml_models.active : %s"%avaml_models.active)


import intentparser
#import verify_subscriber

# todo : vowpal init
