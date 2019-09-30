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
from . import app, logger
from flask import jsonify, abort

@app.errorhandler(415)
def unsupported_mediatype(type=None):
    message = {
            'status': 415,
            'message': 'Unsupported Media Type ' + type,
    }
    response = jsonify(message)
    response.status_code = 415

    return response


def bad_intent_gateway_request():
	"""
	Intent Gateway accepts `text` and `context` fields
	"""
	abort(400, 'Missing `text` parameter in request to Intent Gateway')


def bad_model_context_request():
	"""
	Intent Gateway accepts `text` and `context` fields
	"""
	abort(400, 'Parameter `model_context` needs to be a map (model_type:[models,...])')
