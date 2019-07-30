FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y install bash \
        build-essential \
        wget \
        curl \
        openssh-server \
        git \
        openssh-client && \
    apt-get clean

WORKDIR /root

ENV TF_FORCE_GPU_ALLOW_GROWTH true

RUN pip install tensorflow_datasets Keras-Applications


