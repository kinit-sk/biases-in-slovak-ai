FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04 as base
RUN apt-get update && apt-get install -y curl && apt -y upgrade

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ENV TZ=Europe/Bratislava
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    unzip \
    wget && rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools


# install jupyter lab with extensions and export port
RUN pip3 install jupyterlab==4.0.3
# RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
# RUN apt-get install -y nodejs
# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
# RUN jupyter lab build
EXPOSE 8888
# default shell for jupyter lab
ENV SHELL=/bin/bash

# install pip requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# setup /labs folder for experimental work
RUN mkdir -p /labs && chmod -R a+rwx /labs/
WORKDIR /labs
RUN mkdir /.local && chmod a+rwx /.local

ENV PYTHONPATH "${PYTHONPATH}:/labs/src/"
ENV TRANSFORMERS_CACHE "/labs/cache/transformers"



# This turns off safe directory feature of git, but it is required otherwise the Dockerized git can not work with the repository.
# Without this turned off, WandB can not log git commit hashes.
RUN git config --global --add safe.directory '*'

RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/labs --ip 0.0.0.0 --no-browser --allow-root"]
