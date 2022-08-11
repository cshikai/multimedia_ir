FROM nvcr.io/nvidia/pytorch:22.02-py3


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

WORKDIR /src

ADD requirements.txt /src/requirements.txt

RUN python -m pip install -r requirements.txt

ADD /src /src

# RUN python lumapi/initialize_db.py

# CMD ["uvicorn", "main:api", "--host", "0.0.0.0"]
