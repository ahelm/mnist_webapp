IMAGE_NAME=mnist_webapp

.PHONY: build
build:
		docker build . -t $(IMAGE_NAME)

.PHONY: download
download:
		docker run --rm -it -v $(shell pwd):/usr/src/app $(IMAGE_NAME) download_data.py