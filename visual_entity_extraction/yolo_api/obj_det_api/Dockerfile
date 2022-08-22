FROM nvcr.io/nvidia/pytorch:20.12-py3

WORKDIR /src

ENV DEBIAN_FRONTEND noninteractive

ADD requirements.txt /src/requirements.txt

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg

RUN python -m pip install -r requirements.txt

ADD /src /src

RUN cd /src/yolo/mish-cuda && python setup.py build install


# RUN python lumapi/initialize_db.py

# CMD ["uvicorn", "main:api", "--host", "0.0.0.0"]
