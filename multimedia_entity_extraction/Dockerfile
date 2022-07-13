# For more information, please refer to https://aka.ms/vscode-docker-python

FROM python:3.8

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

ADD build /build
WORKDIR /build
RUN make

ADD /src /src

WORKDIR /src
# During debugging, this entry point will be overridden. For more information, refer to https://aka.ms/vscode-docker-python-debug
