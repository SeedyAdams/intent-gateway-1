help:
	@echo "build - Build container"
	@echo "build-no-cache - Clear cache and deploy container"
	@echo "test - Build container and run tests"

ifeq ($(IMAGE_TAG),)
  ID := "latest"
else
  ID := $(IMAGE_TAG)
endif

build:
	docker build -t="ava/intent-gateway":$(ID) .

build-no-cache:
	docker build -t="ava/intent-gateway":$(ID) .

test: build
	docker run --env RUN_LOCAL_CONTAINER=1 "ava/intent-gateway":$(ID) bash -c 'cd /opt/app; ls -la; /opt/venv/bin/gunicorn --bind 0.0.0.0:80 --daemon IntentGateway:app; cd /opt/app/IntentGateway; /opt/venv/bin/pytest -s -v'
