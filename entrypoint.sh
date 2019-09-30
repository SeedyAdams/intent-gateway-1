#!/bin/bash

function getIGCode() {
    rm -rf $APP_DIR/*
    aws s3 cp --recursive s3://${S3_BUCKET_ZIP_FILE}/docker/ $APP_DIR/
    touch $APP_DIR/*.zip
    mv $APP_DIR/*.zip $APP_DIR/ig_latest.zip
    unzip $APP_DIR/ig_latest.zip -d $APP_DIR/
    rm -rf $APP_DIR/ig_latest.zip
}

if [ "$RUN_LOCAL_CONTAINER" != 1 ]; then
    echo "RUNNING ON ECS!"
    getIGCode

elif ([ "$AWS_ZIP" == 1 ]); then
    echo "RUNNING LOCALLY WITH AWS ZIP!"
    getIGCode

else
    echo "RUNNING LOCALLY WITH VOLUME MOUNT!"
fi

source activate $CONDA_ENV_NAME
cd /opt/app
pip install -r $APP_DIR/requirements_pip.txt
conda install -c conda-forge --yes --file $APP_DIR/conda_requirements.txt
gunicorn --bind 0.0.0.0:5001 IntentGateway:app

