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
import boto3
from . import logger

class AWSClients(object):
    """
    AWS services session handler
    """

    __services = None

    @classmethod
    def init_client(cls, service, local=False):
        """
        initialize aws client
        """
        try:
            if local:
                session = boto3.Session(profile_name=os.environ["PROFILE_NAME"], region_name='us-east-1' )
                return session.client(service)

                # return boto3.client(service,
                #     aws_access_key_id=os.environ['LOCAL_AWS_ACCESS_KEY_ID'],
                #     aws_secret_access_key=os.environ['LOCAL_AWS_SECRET_ACCESS_KEY'],
                #     aws_session_token=os.environ['LOCAL_AWS_SESSION_TOKEN'],
                #     region_name=os.environ["%s_DEFAULT_REGION"%service.upper()])
            else:
                return boto3.client(service,
                    region_name=os.environ["%s_DEFAULT_REGION"%service.upper()])
        except Exception, e:
            return e

    def __new__(cls, services=['kinesis', 's3'], local=False):
        """
        init AWS session handler
        """
        if AWSClients.__services is None:
            AWSClients.__services = object.__new__(cls)
        AWSClients.__services.active = {}
        
        # if (local and not os.environ['LOCAL_AWS_ACCESS_KEY_ID']):
        #     raise Exception("AWSClients :  need AWS credentails for local access. \
        #                 required: LOCAL_AWS_ACCESS_KEY_ID, LOCAL_AWS_SECRET_ACCESS_KEY and LOCAL_AWS_SESSION_TOKEN")
        if (local and not os.environ['PROFILE_NAME']):
            raise Exception("AWSClients :  need AWS credentails for local access. \
                        required: PROFILE Name")

        for service in services:
            AWSClients.__services.active[service] = AWSClients.init_client(service, local)
        return AWSClients.__services

