# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvcr.io/nvidia/tritonserver:20.12-py3


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

CMD ["tritonserver", "--model-repository=/models"]
# RUN apt-get update -y
# RUN apt-get upgrade -y
# RUN pip install --upgrade pip
# RUN apt-get install proj-bin -y
# Install pip requirements


# ADD requirements.txt .
# RUN python -m pip install -r requirements.txt
# RUN pip install jupyter
# RUN python -m pip install "dask[dataframe]"
# RUN pip install fastparquet
# ADD /src /src

#VOLUME ["/src/files"]
#VOLUME /src/files
#ADD /src/files /src/files

# jupyter notebook
# RUN pip install jupyter

# WORKDIR /src
# WORKDIR /src/files
# RUN ["bash"]
# CMD ["python3","index.py"]

# CMD ["python3", "main.py"]
# During debugging, this entry point will be overridden. For more information, refer to https://aka.ms/vscode-docker-python-debug
