ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=20.04
# ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG BUILD_IMAGE=${LINUX_DISTRO}:${DISTRO_VERSION}

FROM ${BUILD_IMAGE} AS build-base

# Use bash as the default shell
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN apt-get install -y wget git

# Download and install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O anaconda.sh
RUN chmod +x anaconda.sh
RUN bash ./anaconda.sh -b -p /opt/anaconda
RUN rm anaconda.sh
# Set Anaconda environment variables
ENV PATH="/opt/anaconda/bin:${PATH}"

# We need these packages to install pytorch
RUN conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# Install these for compiling
RUN apt-get -y install gcc g++

# Install our mlock allocator for pinned memory
WORKDIR /opt
RUN git clone https://github.com/gustrain/mlock.git
WORKDIR /opt/mlock
RUN python setup.py install

# RUN ulimit -l

# pytorch
WORKDIR /opt
RUN git clone https://github.com/mengwanguc/pytorch-meng.git
WORKDIR /opt/pytorch-meng
RUN git submodule update --init --recursive

RUN git checkout gus-emulator-minio

RUN export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} \
    && export USE_CUDA=0 \
    && python setup.py install

# torchvision
WORKDIR /opt
RUN git clone https://github.com/mengwanguc/torchvision-meng.git
WORKDIR /opt/torchvision-meng
RUN git checkout gus-min-io
RUN python setup.py install

# vmtouch
RUN apt install -y build-essential
WORKDIR /opt
RUN git clone https://github.com/hoytech/vmtouch.git
WORKDIR /opt/vmtouch
RUN make
RUN make install

# minio
WORKDIR /opt
RUN git clone https://github.com/gustrain/minio.git
WORKDIR /opt/minio
RUN python setup.py install

WORKDIR /