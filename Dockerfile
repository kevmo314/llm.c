# Specify the correct NVIDIA CUDA image with CUDNN and development tools
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install system packages
RUN apt update && apt install -y \
    git \
    wget \
    build-essential

# Install Miniconda
RUN mkdir -p /root/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh \
    && bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 \
    && rm /root/miniconda3/miniconda.sh \
    && /root/miniconda3/bin/conda init bash

# Setting the PATH environment variable for conda
ENV PATH /root/miniconda3/bin:$PATH

RUN pip install tqdm tiktoken requests datasets transformers

WORKDIR /root

# Clone necessary repositories to the home directory
RUN git clone https://github.com/NVIDIA/cudnn-frontend.git

COPY . /root/llm.c

WORKDIR /root/llm.c

RUN make train_gpt2cu USE_CUDNN=1

ENV HF_TOKEN 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

RUN python dev/data/github_code.py
