include .env

.PHONY: all
all: build run

.PHONY: build
build:
	docker build -t ${CONTAINER_NAME} --platform x86_64 .

.PHONY: run
run:
	docker run -it -v ${CONTAINER_NAME}

.PHONY: deploy
deploy:
	./deploy_gcp.sh

.PHONY: train
train:
	docker run -it \
		-v $(shell pwd)/output:/app/output \
		-v $(shell pwd)/requirements.txt:/app/requirements.txt \
		-v $(shell pwd)/cli.py:/app/cli.py \
		-v $(shell pwd)/images:/app/images \
		${CONTAINER_NAME} python cli.py train

.PHONY: test
test:
	docker run -it \
		-v $(shell pwd)/output:/app/output \
		-v $(shell pwd)/requirements.txt:/app/requirements.txt \
		-v $(shell pwd)/cli.py:/app/cli.py \
		${CONTAINER_NAME} python cli.py test
