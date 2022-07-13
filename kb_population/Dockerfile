FROM python:3.8

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install make

ADD build /build
WORKDIR /build
RUN make

ADD /src /src

WORKDIR /src
