# Text Entity Linking

Linking text entities to known entities in the knowledge base

The text entity linking pipeline consists of 3 services to perform the entity recognition, entity linking, as well as to provide the backend storage for 
the wikipedia knowledge base and index to store 'articles' data.

For the backend service, we will be making use of Elasticsearch, running out of a Docker container. The source code to spin up the Elasticsearch Docker
service is obtained from the Git repository https://github.com/sherifabdlnaby/elastdocker.

For entity recognition, we made use of a Joint Enity-Relation Extration framework known as JEREX, as it has proven to perform quite well on mention extraction, as well
as entity coreferencing/clustering. The initial code is derived from the https://github.com/lavis-nlp/jerex Git repository and the academnic paper on the framework
can be found on the repository as well. The code base for the Jerex API Service is under the directory multimodal-jerex. 

For entity linking, we are making use of BLINK, am entity linking framework that was created by Facebook Research. It uses Wikipedia as the knowledge base for disambiguating
the mentions identified by Jerex. It takes in the mention span, as well as 100 word tokens prior and post the mention span as the context to perform the entity linking. The
initial code is derived from the https://github.com/facebookresearch/BLINK Git repository and the academnic paper on the framework can be found on the repository as well.
The code base for the BLINK API service can be found under the directory BLINK_api.

## Prerequisites:

Before running the services, ensure that you have the following models and requirements to avoid running into errors or missing dependencies 
when running the API services. 

### Elasticsearch Service

From https://github.com/sherifabdlnaby/elastdocker run:

1. Change directory into the text_entity_extraction directory if you are in the main multimedia_ir directory
```
cd text_entity_extraction
```

2. Clone the Repository
```
git clone https://github.com/sherifabdlnaby/elastdocker
```

3. Change directory into the Elastdocker directory
```
cd elastdocker
```

4. Initialize Elasticsearch Keystore and TLS Self-Signed Certificates
```
make setup
```

### Jerex Service

If running training on DocRED, cd into multimodal-jerex and run: 
```
bash ./scripts/fetch_datasets.sh
```

Else, to just run the Jerex API service using the model checkpoints trained by the Jerex team, cd into multimodal-jerex and run:
```
bash ./scripts/fetch_models.sh
```
The configs that the API service will be using is from the text_entity_extraction/multimodal-jerex/configs/docred_joint/test.yaml config file, and
all required configs should already be set, with the default inference compute set to CPU, to set the API service to run on GPU, see the Notes section below.

To use the model checkpoints trained on the DWIE dataset, go to the Google Drive, under the 'DWIE dataset and model/data/models/dwie_joint' folder download the contents,run
```
cd text_entity_extraction/multimodal-jerex
mkdir -p data/models/dwie
```
and add all the downloaded contents into the folder, and change the files paths under the 'model' configs in 'text_entity_extraction/multimodal-jerex/configs/docred_joint/test.yaml' to
the files in the data/models/dwie folder

* check that the requirements.txt file is in the multimodal-jerex folder as well

### BLINK Service



## Docker Services Setup

To run all 

## BLINK_es

# Notes

- Jerex is currently set to run on CPU, to run it on GPU, set the config 'gpus' under 'text_entity_extraction/multimodal-jerex/configs/docred_joint/test.yaml' from [] to [0] or [whatever_gpu_device]
