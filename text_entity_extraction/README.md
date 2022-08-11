# Text Entity Linking

Linking text entities to known entities in the knowledge base

The text entity linking pipeline consists of 3 services to perform the entity recognition, entity linking, as well as to provide the backend storage for 
the wikipedia knowledge base and index to store 'articles' data.

For the backend service, we will be making use of Elasticsearch, running out of a Docker container. The source code to spin up the Elasticsearch Docker
service is obtained from the Git repository https://github.com/sherifabdlnaby/elastdocker

For entity recognition, we made use of a Joint Enity-Relation Extration framework known as JEREX, as it has proven to perform quite well on mention extraction, as well
as entity coreferencing/clustering. The initial code is derived from the https://github.com/lavis-nlp/jerex Git repository and the academnic paper on the framework
can be found on the repository as well.

For entity linking, we are making use of BLINK, am entity linking framework that was created by Facebook Research. It uses Wikipedia as the knowledge base for disambiguating
the mentions identified by Jerex. It takes in the mention span, as well as 100 word tokens prior and post the mention span as the context to perform the entity linking. The
initial code is derived from the https://github.com/facebookresearch/BLINK Git repository and the academnic paper on the framework can be found on the repository as well.

## Prerequisites:

Before running the services, ensure that you have the following models and requirements to avoid running into errors or missing dependencies 
when running the API services. 



## Multimodal docker setup

To run all 

## BLINK_es


