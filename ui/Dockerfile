FROM python:3.8

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install make

ADD build /build
WORKDIR /build
RUN make

ADD /src /src

WORKDIR /src
CMD streamlit run app.py