NAME?=zhuldyzzhan
DATA_PATH?=/hdd

build:
	docker build -t $(NAME) --network=host .

attach:
	docker attach $(NAME)

exec:
	docker exec -it $(NAME) /bin/bash

run:
	docker run --gpus all --rm -it \
	--net=host \
	--ipc=host \
	-v $(DATA_PATH):/hdd \
	-v $(PWD):/workspace \
	--name=$(NAME) \
	$(NAME)
