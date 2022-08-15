FROM nvcr.io/nvidia/pytorch:20.12-py3


# # Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE 1
# # Turns off buffering for easier container logging
# ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install make

ADD build /build
WORKDIR /build
RUN make

ADD /src /src

WORKDIR /src

# CMD ["uvicorn", "main:api", "--host", "0.0.0.0"]