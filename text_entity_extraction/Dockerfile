FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update
RUN apt-get -y install python3-pip vim git
RUN apt-get -y install libfreetype-dev libfreetype6 libfreetype6-dev

RUN pip install -U pip
COPY haystack .
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fastapi pandas requests torch fsspec && pip install "uvicorn[standard]"

RUN mkdir /BLINK_api && mkdir /BLINK_api/src && mkdir /BLINK_api/models && mkdir /BLINK_api/configs && mkdir /BLINK_api/logs
COPY BLINK_api/src/ /BLINK_api/src
WORKDIR /BLINK_api/src
RUN pip install git+https://github.com/deepset-ai/haystack.git#egg=farm-haystack[colab]

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

CMD ["/bin/bash"]