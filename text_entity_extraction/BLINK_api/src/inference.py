import BLINK_es 
import blink.ner as NER
import blink.candidate_ranking.utils as utils
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
from blink.crossencoder.train_cross import modify
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.biencoder.biencoder import load_biencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)

import argparse
import pathlib
import time
import ast
import json
import os

from hydra.utils import to_absolute_path
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import numpy as np

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever

def config_to_abs_paths(config, *parameter_names):
    absolute_path = pathlib.Path(__file__).parent.resolve()
    absolute_path = pathlib.Path(absolute_path).parent.resolve()
    for param_name in parameter_names:
        param = getattr(config, param_name)
        if param is not None:
            setattr(config, param_name, os.path.join(absolute_path,param))


initialize(config_path="../configs", job_name="blink")
cfg = compose(config_name="blink")
print(OmegaConf.to_yaml(cfg))

config_to_abs_paths(cfg.model, 'biencoder_model')
config_to_abs_paths(cfg.model, 'biencoder_config')
config_to_abs_paths(cfg.model, 'crossencoder_model')
config_to_abs_paths(cfg.model, 'crossencoder_config')
config_to_abs_paths(cfg.logging, 'output_path')

args = cfg

print(args)

def inferenceWrapper(cls):
      
    class Wrapper:
          
        def __init__(self, mentions_to_link):

            self.args = args

            start = time.time()
            logger = utils.get_logger(args.logging.output_path)
            self.models = BLINK_es.load_models(args, logger)
            end = time.time()

            print("Time to load BLINK models",end - start)

            self.wrap = cls(mentions_to_link)
              
        def run_inference(self):

            start = time.time()

            document_store = ElasticsearchDocumentStore(host=args.elasticsearch.host,
                                                port=args.elasticsearch.port, 
                                                username=args.elasticsearch.username, 
                                                password=args.elasticsearch.password, 
                                                scheme=args.elasticsearch.scheme, 
                                                verify_certs=args.elasticsearch.verify_certs, 
                                                index = args.elasticsearch.index_name,
                                                embedding_dim=args.elasticsearch.embedding_dim,
                                                return_embedding=args.elasticsearch.return_embedding,
                                                search_fields=list(args.elasticsearch.search_fields))

            retriever = BM25Retriever(document_store=document_store)

            documents = []

            if len(self.wrap.mentions_to_link) >0:

                print("Total number of mentions to link: ", len(self.wrap.mentions_to_link))

                mentions_to_blink = {
                    "ids":[],
                    "mentions":[]
                }
                mentions_to_exact_match = {
                    "ids":[],
                    "mentions":[]
                }

                for mention in self.wrap.mentions_to_link:
                    mention_length = mention["mention"].split(' ')
                    mention_length = [part for part in mention_length if len(part)>0]

                    print("mention: ", mention["mention"].capitalize())

                    documents = retriever.retrieve(query=mention["mention"].capitalize(),
                                                   top_k=5,
                                                  )

                    document_titles = [document.meta['title'].lower() for document in documents]

                    if len(documents) > 0 and mention["mention"].lower() in document_titles and len(mention_length) > 1:
                        print("Exact matched: ", mention["mention"])
                        mentions_to_exact_match["ids"].append(mention["id"])
                        mention["entity_linked"] = mention["mention"].title()
                        mention["entity_link"] = documents[0].meta['idx']
                        mentions_to_exact_match["mentions"].append(mention)
                    else:
                        # print("To BLINK infer: ", mention["mention"])
                        mentions_to_blink["ids"].append(mention["id"])
                        mentions_to_blink["mentions"].append(mention)

                print("Total number of mentions linked by exact match: ", len(mentions_to_exact_match["ids"]))
                print("Toal number of mentions to be linked by BLINK: ",len(mentions_to_blink["ids"]))

                predictions = []
                links = []
                scores = []
                embeddings = []

                for mention in mentions_to_blink["mentions"]:

                    try:
                        _, _, _, _, _, mention_predictions, mention_links, mention_scores, mention_embeddings = BLINK_es.run(self.args, None, *self.models, test_data=[mention])
                        predictions.extend(mention_predictions)
                        links.extend(mention_links)
                        scores.extend(mention_scores)
                        embeddings.extend(mention_embeddings)
                        error = False
                    except:
                        print("Error while performing entity linking on mention")
                        predictions.append(['Not Linked'])
                        links.append(['None'])
                        scores.append([0.00])
                        embeddings.append(torch.zeros(1024).detach().cpu().numpy())
                        error = True
                        break
            else:
                error = True
            
            end = time.time()
            print("Time to complete entity linking",end - start)

            entities_dict ={}
            if not error:
                ent_list = []
                for i in range(0,len(self.wrap.mentions_to_link)):
                    ent_dict = {}
                    print("Mention: ",self.wrap.mentions_to_link[i]["mention"])
                    print("Original sentence: ")
                    print(self.wrap.mentions_to_link[i]["context_left"] + " " + self.wrap.mentions_to_link[i]["mention"] + " " + self.wrap.mentions_to_link[i]["context_right"])

                    if self.wrap.mentions_to_link[i]["id"] in mentions_to_blink["ids"]:
                        index = mentions_to_blink["ids"].index(self.wrap.mentions_to_link[i]["id"])
                        print("Entity linked: ", predictions[index][0])
                        print("Entities identified: ", predictions[index])
                        print("Score: ", scores[index][0])
                        print("\n")

                        if scores[index][0] > -2.1:

                            ent_dict['doc_id'] = self.wrap.mentions_to_link[i]["doc_id"]
                            ent_dict['mention'] = self.wrap.mentions_to_link[i]["mention"]
                            ent_dict['entity_linked'] = predictions[index][0]
                            ent_dict['entity_link'] = links[index][0]
                            ent_dict['entity_confidence_score'] = scores[index][0]
                            ent_dict['link_type'] = 'BLINK'
                            ent_dict['embeddings'] = torch.zeros(1024).detach().cpu().numpy()

                        else:
                            print("mention: " ,self.wrap.mentions_to_link[i]["mention"])
                            print("prediction: " ,predictions[index][0])
                            print("score: " ,scores[index][0])
                            ent_dict['doc_id'] = self.wrap.mentions_to_link[i]["doc_id"]
                            ent_dict['mention'] = self.wrap.mentions_to_link[i]["mention"]
                            ent_dict['entity_linked'] = "Unknown"
                            ent_dict['entity_link'] = "Unknown"
                            ent_dict['entity_confidence_score'] = 0.0
                            ent_dict['link_type'] = 'BLINK'
                            ent_dict['embeddings'] = embeddings[index]
                        
                        ent_list.append(ent_dict)
                    else:
                        index = mentions_to_exact_match["ids"].index(self.wrap.mentions_to_link[i]["id"])
                        ent_dict['doc_id'] = self.wrap.mentions_to_link[i]["doc_id"]
                        ent_dict['mention'] = self.wrap.mentions_to_link[i]["mention"]
                        ent_dict['entity_linked'] = mentions_to_exact_match["mentions"][index]['entity_linked']
                        ent_dict['entity_link'] = mentions_to_exact_match["mentions"][index]['entity_link']
                        ent_dict['entity_confidence_score'] = 1.0
                        ent_dict['link_type'] = 'Exact'
                        ent_dict['embeddings'] = torch.zeros(1024).detach().cpu().numpy()
                        ent_list.append(ent_dict)

                entities_dict['entities'] = ent_list

            return entities_dict
          
    return Wrapper

def encode_candidate(
    reranker,
    candidate_pool,
    encode_batch_size,
    silent,
    logger,
):
    reranker.model.eval()
    device = reranker.device
    #for cand_pool in candidate_pool:
    #logger.info("Encoding candidate pool %s" % src)
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


def load_candidate_pool(
    tokenizer,
    params,
    logger,
    cand_pool_path,
):
    candidate_pool = None
    # try to load candidate pool from file
    try:
        logger.info("Loading pre-generated candidate pool from: ")
        logger.info(cand_pool_path)
        candidate_pool = torch.load(cand_pool_path)
    except:
        logger.info("Loading failed.")
    assert candidate_pool is not None

    return candidate_pool

def KBWrapper(cls):
      
    class Wrapper:
          
        def __init__(self, entities_to_add):

            self.args = args

            self.wrap = cls(entities_to_add)

            self.document_store = ElasticsearchDocumentStore(host=args.elasticsearch.host,
                                                             port=args.elasticsearch.port, 
                                                             username=args.elasticsearch.username, 
                                                             password=args.elasticsearch.password, 
                                                             scheme=args.elasticsearch.scheme, 
                                                             verify_certs=args.elasticsearch.verify_certs, 
                                                             index = args.elasticsearch.index_name,
                                                             embedding_dim=args.elasticsearch.embedding_dim)

        def add_to_jsonl_kb(self,new_entities_list):

            json_list = []
            with open(self.args.entity_catalogue, "r") as fin:
                    lines = fin.readlines()
                    for line in lines:
                        entity = json.loads(line)
                        json_list.append(entity)

            json_list.extend(new_entities_list)

            with open(self.args.new_entity_catalogue, 'w') as outfile:
                for entry in json_list:
                    json.dump(entry, outfile)
                    outfile.write('\n')

            print("Done adding new entities")

            return new_entities_list

        def add_to_elasticsearch_kb(self,new_entities_list,wikipedia_embeddings):

            docs = []

            for index in tqdm(range(0,len(new_entities_list))):
                doc = {}
                doc['content'] = new_entities_list[index]['text']
                doc['meta'] = {'idx':new_entities_list[index]['idx'],'title':new_entities_list[index]['title'], 'entity': new_entities_list[index]['entity']}
                doc['embedding'] = wikipedia_embeddings[index].detach().cpu().numpy()

                docs.append(doc)

            self.document_store.write_documents(docs)

            print("Done adding new entities")

            return new_entities_list

        def generate_biencoder_token_ids(self,entities):

            with open(self.args.model.biencoder_config) as json_file:
                biencoder_params = json.load(json_file)
                biencoder_params["path_to_model"] = self.args.model.biencoder_model
            biencoder = load_biencoder(biencoder_params)

            print(entities)

            print("Generating token_ids")

            # Get token_ids corresponding to candidate title and description
            tokenizer = biencoder.tokenizer
            max_context_length, max_cand_length =  biencoder_params["max_context_length"], biencoder_params["max_cand_length"]
            max_seq_length = max_cand_length
            ids = []

            for entity in entities:
                candidate_desc = entity['text']
                candidate_title = entity['title']
                cand_tokens = get_candidate_representation(
                    candidate_desc, 
                    tokenizer, 
                    max_seq_length, 
                    candidate_title=candidate_title
                )

                token_ids = cand_tokens["ids"]
                ids.append(token_ids)

            ids = torch.tensor(ids)

            return ids

        def generate_candidates(self,biencoder_ids):

            with open(self.args.model.biencoder_config) as json_file:
                biencoder_params = json.load(json_file)
                biencoder_params["path_to_model"] = self.args.model.biencoder_model
            
            # biencoder_params["entity_dict_path"] = self.args.entities_to_add
            biencoder_params["data_parallel"] = True
            biencoder_params["no_cuda"] = False
            biencoder_params["max_context_length"] = 32
            biencoder_params["encode_batch_size"] = 8

            biencoder = load_biencoder(biencoder_params)

            logger = utils.get_logger(biencoder_params.get("model_output_path", None))

            # candidate_pool = load_candidate_pool(
            #     biencoder.tokenizer,
            #     biencoder_params,
            #     logger,
            #     getattr(args, 'saved_cand_ids', None),
            # )

            candidate_pool = biencoder_ids

            print(candidate_pool.shape)
      
            candidate_encoding = encode_candidate(
                biencoder,
                candidate_pool,
                biencoder_params["encode_batch_size"],
                biencoder_params["silent"],
                logger,
            )
            
            print(candidate_encoding.shape)

            print(candidate_encoding[0,:10])

            return candidate_encoding

        def merge_with_original_embeddings(self,candidate_encoding):
            all_chunks = []

            original_embeddings_path = self.args.entity_encoding

            if not os.path.exists(original_embeddings_path) or os.path.getsize(original_embeddings_path) == 0:
                print("Path to orignial embeddings incorrect or original embeddings file is empty")

            print("Loading original embeddings!!!")

            try:
                loaded_chunk = torch.load(original_embeddings_path)
                print("Initial number of embeddings: ",loaded_chunk.shape[0])
            except:
                print("Path to orignial embeddings incorrect or unable to load torch embeddings from file path {}".format(original_embeddings_path))

            all_chunks.append(loaded_chunk)

            all_chunks.append(candidate_encoding)

            all_chunks = torch.cat(all_chunks, dim=0)

            print(all_chunks.shape)

            del loaded_chunk

            torch.save(all_chunks, 'models/test_candidate_embeddings.t7')

            print("Saved in 'models/test_candidate_embeddings.t7' ")

            return all_chunks

        def create_faiss_index(self,candidate_encoding):
            output_path = self.args.faiss_output_path
            output_dir, _ = os.path.split(output_path)
            logger = utils.get_logger(output_dir)

            vector_size = candidate_encoding.size(1)
            index_buffer = 50000

            if True:
                logger.info("Using HNSW index in FAISS")
                index = DenseHNSWFlatIndexer(vector_size, index_buffer)
            else:
                logger.info("Using Flat index in FAISS")
                index = DenseFlatIndexer(vector_size, index_buffer)

            logger.info("Building index.")
            index.index_data(candidate_encoding.numpy())
            logger.info("Done indexing data.")

            index.serialize(output_path)

            return

        def add_entities_to_kb(self):

            # print("Runnning add_to_jsonl_kb!!!")
            # new_entities = self.add_to_jsonl_kb(self.wrap.entities_to_add)
            
            print("Runnning generate_biencoder_token_ids!!!")
            biencoder_ids = self.generate_biencoder_token_ids(self.wrap.entities_to_add)
            torch.cuda.empty_cache()
            print("Runnning generate_candidates!!!")
            candidate_embedding = self.generate_candidates(biencoder_ids)
            torch.cuda.empty_cache()
            print("Runnning add_to_elasticsearch_kb!!!")
            new_entities = self.add_to_elasticsearch_kb(self.wrap.entities_to_add,candidate_embedding)
            torch.cuda.empty_cache()

            # print("Runnning generate_candidates!!!")
            # candidate_embedding = self.generate_candidates(biencoder_ids)
            # torch.cuda.empty_cache()
            # print("Runnning merge_with_original_embeddings!!!")
            # full_candidate_embedding = self.merge_with_original_embeddings(candidate_embedding)
            # torch.cuda.empty_cache()
            # print("Runnning create_faiss_index!!!")
            # self.create_faiss_index(full_candidate_embedding)
            # print("Done!")

            return
          
    return Wrapper
  
@inferenceWrapper
class Inference:
    def __init__(self, mentions_to_link):

        # mentions_to_link = ast.literal_eval(mentions_to_link)

        for mention in mentions_to_link:
            if "label" not in mention.keys():
                mention["label"] = "unknown"
            if "label" not in mention.keys():
                mention["label_id"] = -1
            if not (set(["context_left","mention","context_right"]).issubset(set(list(mention.keys())))):
                print("Mention dictionary does not contain 'context_left','mention', or 'context_right' field, will result in error when running inference.")

        self.mentions_to_link = mentions_to_link

@KBWrapper
class NewKBEntities:
    def __init__(self, entities_to_add):

        
        # [{"text": " Shearman Chua was born in Singapore, in the year 1996. He is an alumnus of NTU and is currently working at DSTA. ", "idx": "https://en.wikipedia.org/wiki?curid=88767376", "title": "Shearman Chua", "entity": "Shearman Chua"},
        # {"text": " The COVID-19 recession is a global economic recession caused by the COVID-19 pandemic. The recession began in most countries in February 2020. After a year of global economic slowdown that saw stagnation of economic growth and consumer activity, the COVID-19 lockdowns and other precautions taken in early 2020 drove the global economy into crisis. Within seven months, every advanced economy had fallen to recession. The first major sign of recession was the 2020 stock market crash, which saw major indices drop 20 to 30% in late February and March. Recovery began in early April 2020, as of April 2022, the GDP for most major economies has either returned to or exceeded pre-pandemic levels and many market indices recovered or even set new records by late 2020. ", "idx": "https://en.wikipedia.org/wiki?curid=63462234", "title": "COVID-19 recession", "entity": "COVID-19 recession"},
        # {"text": " The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The novel virus was first identified from an outbreak in Wuhan, China, in December 2019. Attempts to contain it there failed, allowing the virus to spread worldwide. The World Health Organization (WHO) declared a Public Health Emergency of International Concern on 30 January 2020 and a pandemic on 11 March 2020. As of 15 April 2022, the pandemic had caused more than 502 million cases and 6.19 million deaths, making it one of the deadliest in history. ", "idx": "https://en.wikipedia.org/wiki?curid=62750956", "title": "COVID-19 pandemic", "entity": "COVID-19 pandemic"}]

        self.entities_to_add = entities_to_add