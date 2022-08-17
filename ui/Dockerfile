FROM python:3.8.2-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install make

WORKDIR /ui
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY ./app.py .
CMD streamlit run app.py
