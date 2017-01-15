DEBUG:=1
VERSION:=v0.1
IMAGE_TAG:= icyrizard/eos.git:$(VERSION)
DOCKER_FILE:= Dockerfile-dev
HERE:=$(shell pwd)
DOCKER_RUN_FLAGS:= --rm \
	--volume $(HERE)/:/eos/ \
	--volume $(HERE)/examples/data/:/data/ \
	--volume $(HERE)/share/:/data/share/ \
	--volume ~/.bash_history:/root/.bash_history \ -e "DEBUG=$(DEBUG)"

BASE_DOCKER_CMD:= docker run $(DOCKER_RUN_FLAGS) $(IMAGE_TAG)

$(info $(TARGETS))

TARGETS:= targets/fit-model \
	targets/fit-model-simple \
	targets/fit-model-multi-frame

all: $(DEPENDENCIES) $(TARGETS)

OS := $(shell uname)

.PHONY: build
build:
	docker build -f $(DOCKER_FILE) -t $(IMAGE_TAG) .

.PHONY: run-bash
run-bash:
	docker run --interactive --tty $(DOCKER_RUN_FLAGS) $(IMAGE_TAG) /bin/bash

.PHONY: run-bash-cmd
run-bash-cmd:
	docker run --interactive --tty $(DOCKER_RUN_FLAGS) $(IMAGE_TAG) \
		/bin/bash -c "$(CMD)"

# TODO create pkg files for the include libraries.
targets/fit-model-multi-frame: examples/fit-model-multi-frame.cpp
	mkdir -p $(dir $@)
	$(BASE_DOCKER_CMD) /bin/bash -c \
		'clang++ -O3 -std=c++14 \
			-ggdb \
			-I/usr/local/include/pybind11/include/ \
			-I/eos/include/ \
			-I/eos/3rdparty/glm/ \
			-I/eos/3rdparty/eigen3-nnls/ \
			-I/eos/3rdparty/cereal-1.1.1/include/ \
			-I/eos/3rdparty/eigen3-nnls/src \
			-I/eos/3rdparty/nanoflann/include \
			-I/usr/local/include/opencv2/ \
			-I/usr/include/boost/ \
			-I/usr/include/eigen3/ \
			-L/usr/lib/x86_64-linux-gnu/ \
			-L/usr/local/lib/ \
			-lboost_program_options \
			-lboost_filesystem \
			-lopencv_world \
			-lpthread \
			-lboost_system \
		$^ -o $@'

targets/fit-model-simple: examples/fit-model-simple.cpp
	mkdir -p $(dir $@)
	$(BASE_DOCKER_CMD) /bin/bash -c \
		'clang++ -O3 -std=c++14 \
			-ggdb \
			-I/usr/local/include/pybind11/include/ \
			-I/eos/include/ \
			-I/eos/3rdparty/glm/ \
			-I/eos/3rdparty/eigen3-nnls/ \
			-I/eos/3rdparty/cereal-1.1.1/include/ \
			-I/eos/3rdparty/eigen3-nnls/src \
			-I/eos/3rdparty/nanoflann/include \
			-I/usr/local/include/opencv2/ \
			-I/usr/include/boost/ \
			-I/usr/include/eigen3/ \
			-L/usr/lib/x86_64-linux-gnu/ \
			-L/usr/local/lib/ \
			-lboost_program_options \
			-lboost_filesystem \
			-lopencv_world \
			-lpthread \
			-lboost_system \
		$^ -o $@'

targets/fit-model: examples/fit-model.cpp
	mkdir -p $(dir $@)
	$(BASE_DOCKER_CMD) /bin/bash -c \
		'clang++ -ggdb -O3 -std=c++14 \
			-I/usr/local/include/pybind11/include/ \
			-I/eos/include/ \
			-I/eos/3rdparty/glm/ \
			-I/eos/3rdparty/eigen3-nnls/ \
			-I/eos/3rdparty/cereal-1.1.1/include/ \
			-I/eos/3rdparty/eigen3-nnls/src \
			-I/eos/3rdparty/nanoflann/include \
			-I/usr/local/include/opencv2/ \
			-I/usr/include/boost/ \
			-I/usr/include/eigen3/ \
			-L/usr/lib/x86_64-linux-gnu/ \
			-L/usr/local/lib/ \
			-lboost_program_options \
			-lboost_filesystem \
			-lopencv_world \
			-lpthread \
		$^ -o $@'
