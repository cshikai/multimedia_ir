# FROM nvcr.io/nvidia/pytorch:20.12-py3
FROM nvcr.io/nvidia/pytorch:22.01-py3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ADD build /build
WORKDIR /build
RUN make

ADD /src /src
RUN mkdir /data
RUN mkdir /models

WORKDIR /src
