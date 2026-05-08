# Use a lightweight Python 3.12 base image
FROM python:3.12-slim

# Set up working directory
ENV LANG=C.UTF-8
ENV SHELL=/bin/bash

# Install dependencies needed for uv
RUN apt-get update && apt-get install -y curl unzip && \
    curl -Ls https://astral.sh/uv/install.sh | sh 

# RUN apt update && install build-essential && export CC=/usr/bin/gcc