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
import logging

stdout_logger =  os.environ["LOG_STDOUT"] if "LOG_STDOUT" in os.environ and len(os.environ["LOG_STDOUT"]) else False

class Logger(logging.getLoggerClass()):

    def __init__(self, logger_name='IntentGateway',
                    filename='/var/log/tmp.log',
                    kinesis_stream_name=None,
                    log_kinesis=False,
                    level=logging.ERROR):
        """
        logger object with support for file and stream based logging
        """
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler = logging.FileHandler(filename)
        self.handler.setLevel(level)
        self.handler.setFormatter(self.formatter)

        self.stdout_handler = logging.StreamHandler(sys.stdout)
        self.stdout_handler.setLevel(level)
        self.stdout_handler.setFormatter(self.formatter)

        self.logger = logging.getLogger(logger_name)
        self.logger.addHandler(self.handler)

        # debugging only
        if stdout_logger:
            self.logger.addHandler(self.stdout_handler)

        self.logger.setLevel(level)
        self.log_kinesis = log_kinesis

    def info(self, message):
        """
        log_level : 20/INFO
        """
        self.logger.info(message)
        if self.log_kinesis:
            # todo : setup kinesis logging with session_id's
            # self.logger.info(message*2)
            pass

    def warn(self, message):
        """
        log_level : 30/WARN
        """
        self.logger.warn(message)
        if self.log_kinesis:
            # todo : setup kinesis logging with session_id's
            # self.logger.info(message*2)
            pass

    def error(self, message):
        """
        log_level : 40/WARN
        """
        self.logger.error(message)
        if self.log_kinesis:
            # todo : setup kinesis logging with session_id's
            # self.logger.info(message*2)
            pass
