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
duckparser API
"""
import json
import jpype
from duckling import DucklingWrapper

from . import app, logger

d = {}

def init_duck():
    "cache duckling parser"
    if 'duck' not in d:
        logger.info("re/initializing parser, JVMStarted_state = %d"%jpype.isJVMStarted())
        duck = DucklingWrapper()
        d['duck'] = duck
        logger.info("duck ready to parse....")


@app.route("/intentgateway/v1/duck-date")
def duck_date_demo():
    logger.info("all good!")
    return "all good!"


@app.route("/intentgateway/v1/test_date_parse")
def get_date():
    init_duck()
    parsed_date = d['duck'].parse_time(u'i filed my claim on 12/25/2016')
    return parsed_date[0]['value']['value']


@app.route("/intentgateway/v1/extract-datetime")
def extract_datetime(user_text):
    "returns datetime in ISO-8601 format"
    init_duck()
    parsed_dates = d['duck'].parse_time(user_text)
    date_ids = {}
    for date in parsed_dates:
        date_ids[date['text']] = date['value']['value']
    return date_ids
