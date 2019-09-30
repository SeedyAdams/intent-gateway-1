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

FROM conda/miniconda2
MAINTAINER Cory Adams <cory.adams@asurion.com>

# Install Updates and Essentials
RUN apt-get clean \
	&& apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y \
	libpq-dev \
	build-essential \
	gcc \
	wget \
	curl \
	unzip \
	&& pip install --upgrade pip \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install AWS CLI
RUN pip install --user --upgrade awscli

# Configure Environment
ENV APP_DIR /opt/app/IntentGateway
ENV ENV_DIR /opt/app/env
ENV CONDA_ENV_NAME ig_py2
ENV SHELL /bin/bash
ENV PYTHON_VERSION 2.7
ENV CONDA_ENV_PATH /usr/local/envs/$CONDA_ENV_NAME/bin

# Creating Project directory, installing new conda environment
RUN mkdir -p $APP_DIR \
    && mkdir -p $ENV_DIR \
    && conda update conda -y \
    && conda create -y -n $CONDA_ENV_NAME python=$PYTHON_VERSION pip

#conda update -n base -c defaults conda -y

# Export Path Variable for Environment
ENV PATH /opt/conda/envs/$CONDA_ENV_NAME/bin:/root/.local/bin:$PATH

# Copy package requirement files
COPY ./requirements_pip.txt $APP_DIR/requirements_pip.txt
COPY ./conda_requirements.txt $APP_DIR/conda_requirements.txt

# Activate Virtual Environment and install dependencies
RUN $SHELL -c "source activate $CONDA_ENV_NAME && conda install -c conda-forge --yes --file $APP_DIR/conda_requirements.txt"

RUN $CONDA_ENV_PATH/pip install --upgrade pip \
    && $CONDA_ENV_PATH/pip install -r $APP_DIR/requirements_pip.txt \
    && $CONDA_ENV_PATH/python -m spacy download en \
    && $CONDA_ENV_PATH/python -m nltk.downloader snowball_data punkt treebank

COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
RUN touch /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

#ADD . $APP_DIR