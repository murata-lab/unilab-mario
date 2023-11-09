IMAGE_NAME := mario
CONTAINER_NAME := mario

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p 8888:8888 --name $(CONTAINER_NAME) $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)