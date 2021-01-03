IMAGE_NAME=mnist_webapp
CONTAINER_NAME=$(IMAGE_NAME)

.PHONY: serve
serve: data/mnist_model.pt
		docker run --rm -it -v $(shell pwd):/usr/src/app \
			--platform linux/amd64 \
			--entrypoint "./run_server.sh" \
			--name "$(IMAGE_NAME)_serve" \
			-p 8000:8000 \
			$(IMAGE_NAME)

.PHONY: build
build:
		docker buildx build . -t $(IMAGE_NAME) --platform linux/amd64

data/mnist_model.pt: train

.PHONY: train
train: build
		docker run --rm -it -v $(shell pwd):/usr/src/app \
			--platform linux/amd64 \
			--name "$(IMAGE_NAME)_train" \
			$(IMAGE_NAME) \
			mnist_model.py

.PHONY: pngs
pngs:
		docker run --rm -it -v $(shell pwd):/usr/src/app \
			--platform linux/amd64 \
			--name "$(IMAGE_NAME)_png" \
			$(IMAGE_NAME) \
			generate_pngs.py

.PHONY: freeze
freeze:
		.venv/bin/pip freeze > requirements.txt

.PHONY: clean
clean:
		docker image rm $(IMAGE_NAME)

.PHONY: clean-all
clean-all: clean
		rm -rf data
