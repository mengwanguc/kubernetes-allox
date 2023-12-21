FROM wangm12/gpemu-pytorch:base

WORKDIR /opt/pytorch-meng
RUN git checkout original-pytorch
RUN export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} \
    && export USE_CUDA=0 \
    && python setup.py install

WORKDIR /opt/torchvision-meng
RUN git checkout original-torchvision
RUN python setup.py install

WORKDIR /app

COPY . .
RUN chmod +x self-check.sh

