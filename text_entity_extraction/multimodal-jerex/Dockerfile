FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN mkdir -p /jerex
WORKDIR /jerex

COPY . /jerex

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install jupyter
RUN pip install hydra-core --upgrade
RUN pip install fastapi pandas requests && pip install yamlargparse==1.4.0 && pip install pyyaml && pip install "uvicorn[standard]"
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
