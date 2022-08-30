import argparse
import pathlib
import json
import sys
import os
import time

from tqdm import tqdm
import logging
import torch
import numpy as np

from hydra.utils import to_absolute_path
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.crossencoder.train_cross import modify, evaluate
from blink.crossencoder.data_process import prepare_crossencoder_data

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever

# def config_to_abs_paths(config, *parameter_names):
#     absolute_path = pathlib.Path(__file__).parent.resolve()
#     absolute_path = pathlib.Path(absolute_path).parent.resolve()
#     for param_name in parameter_names:
#         param = getattr(config, param_name)
#         if param is not None:
#             setattr(config, param_name, os.path.join(absolute_path,param))


# initialize(config_path="../configs", job_name="blink")
# cfg = compose(config_name="blink")
# print(OmegaConf.to_yaml(cfg))

# config_to_abs_paths(cfg.model, 'biencoder_model')
# config_to_abs_paths(cfg.model, 'biencoder_config')
# config_to_abs_paths(cfg.model, 'crossencoder_model')
# config_to_abs_paths(cfg.model, 'crossencoder_config')
# config_to_abs_paths(cfg.logging, 'output_path')

# args = cfg

# print(args)

def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    embeddings = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(
                    context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores, batch_embeddings = biencoder.score_candidate(
                    # .to(device)
                    context_input, None, cand_encs=candidate_encoding
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()
                batch_embeddings = batch_embeddings.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
        embeddings.append(batch_embeddings)
    return labels, nns, all_scores, embeddings


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger,
                   context_len, zeshel=False, silent=False)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits


def load_models(args, logger=None):
    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.model.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.model.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.model.crossencoder_config) as json_file:
            crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.model.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params
    )


def _load_wiki(
        args,
        logger=None):

    # load all the wikipedia entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0

    document_store = ElasticsearchDocumentStore(host=args.elasticsearch.host,
                                                port=args.elasticsearch.port,
                                                username=args.elasticsearch.username,
                                                password=args.elasticsearch.password,
                                                scheme=args.elasticsearch.scheme,
                                                verify_certs=args.elasticsearch.verify_certs,
                                                index=args.elasticsearch.index_name,
                                                embedding_dim=args.elasticsearch.embedding_dim)

    start_time = time.time()

    docs = document_store.get_all_documents(batch_size=10000)

    print("time taken to load wikipedia documents" %
          (time.time() - start_time))

    for doc in docs:
        entity = doc.meta
        entity['text'] = doc.content

        if "idx" in entity:
            split = entity["idx"].split("curid=")
            if len(split) > 1:
                wikipedia_id = int(split[-1].strip())
            else:
                wikipedia_id = entity["idx"].strip()

            assert wikipedia_id not in wikipedia_id2local_id
            wikipedia_id2local_id[wikipedia_id] = local_idx

        title2id[entity["title"]] = local_idx
        id2title[local_idx] = entity["title"]
        id2text[local_idx] = entity["text"]
        local_idx += 1
    return (
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id
    )


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    test_data=None,
):

    print("running BLINK es!!!")

    samples = test_data

    # don't look at labels
    keep_all = (
        samples[0]["label"] == "unknown"
        or samples[0]["label_id"] < 0
    )

    document_store = ElasticsearchDocumentStore(host=args.elasticsearch.host,
                                                port=args.elasticsearch.port,
                                                username=args.elasticsearch.username,
                                                password=args.elasticsearch.password,
                                                scheme=args.elasticsearch.scheme,
                                                verify_certs=args.elasticsearch.verify_certs,
                                                index=args.elasticsearch.index_name,
                                                embedding_dim=args.elasticsearch.embedding_dim,
                                                return_embedding=args.elasticsearch.return_embedding,
                                                search_fields=list(args.elasticsearch.search_fields))

    retriever = BM25Retriever(document_store=document_store)

    documents = []

    for sample in samples:
        candidate_documents = retriever.retrieve(query=sample['mention'],
                                                 top_k=args.elastic_candidates
                                                 )
        documents.extend(candidate_documents)

    if len(documents) < 1:

        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        embeddings = []
        for batch in tqdm(dataloader):
            context_input, _, label_ids = batch
            with torch.no_grad():

                batch_embeddings = biencoder.encode_batch(context_input)
                batch_embeddings = batch_embeddings.data.numpy()

            embeddings.append(batch_embeddings)

        return (
            -1,
            -1,
            -1,
            -1,
            len(samples),
            [['None' for sample in samples]],
            [['None' for sample in samples]],
            [[-5.0 for sample in samples]],
            embeddings
        )

    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    candidate_encoding = []
    local_idx = 0

    for doc in documents:
        entity = doc.meta
        print("Wikipedia entity retrieved: ", entity['title'])
        entity['text'] = doc.content

        if "idx" in entity:
            split = entity["idx"].split("curid=")
            if len(split) > 1:
                wikipedia_id = int(split[-1].strip())
            else:
                wikipedia_id = entity["idx"].strip()

            if wikipedia_id not in wikipedia_id2local_id:
                wikipedia_id2local_id[wikipedia_id] = local_idx
                title2id[entity["title"]] = local_idx
                id2title[local_idx] = entity["title"]
                id2text[local_idx] = entity["text"]
                candidate_encoding.append(torch.from_numpy(doc.embedding))
                local_idx += 1

    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )

    top_k = args.top_k if len(
        candidate_encoding) > args.top_k else len(candidate_encoding)

    candidate_encoding = torch.stack((candidate_encoding))

    labels, nns, scores, embeddings = _run_biencoder(
        biencoder, dataloader, candidate_encoding, top_k, None
    )

    biencoder_accuracy = -1
    recall_at = -1
    if not keep_all:
        # get recall values
        top_k = args.top_k
        x = []
        y = []
        for i in range(1, top_k):
            temp_y = 0.0
            for label, top in zip(labels, nns):
                if label in top[:i]:
                    temp_y += 1
            if len(labels) > 0:
                temp_y /= len(labels)
            x.append(i)
            y.append(temp_y)
        # plt.plot(x, y)
        biencoder_accuracy = y[0]
        recall_at = y[-1]
        print("biencoder accuracy: %.4f" % biencoder_accuracy)
        print("biencoder recall@%d: %.4f" % (top_k, y[-1]))

    if args.fast:

        predictions = []
        for entity_list in nns:
            sample_prediction = []
            for e_id in entity_list:
                e_title = id2title[e_id]
                sample_prediction.append(e_title)
            predictions.append(sample_prediction)

        # use only biencoder
        return (
            biencoder_accuracy,
            recall_at,
            -1,
            -1,
            len(samples),
            predictions,
            scores,
            embeddings,
        )

    # prepare crossencoder data
    context_input, candidate_input, label_input = prepare_crossencoder_data(
        crossencoder.tokenizer, samples, labels, nns, id2title, id2text, keep_all,
    )

    context_input = modify(
        context_input, candidate_input, crossencoder_params["max_seq_length"]
    )

    dataloader = _process_crossencoder_dataloader(
        context_input, label_input, crossencoder_params
    )

    # run crossencoder and get accuracy
    accuracy, index_array, unsorted_scores = _run_crossencoder(
        crossencoder,
        dataloader,
        logger,
        context_len=biencoder_params["max_context_length"],
    )

    scores = []
    predictions = []
    links = []
    for entity_list, index_list, scores_list in zip(
        nns, index_array, unsorted_scores
    ):

        index_list = index_list.tolist()

        # descending order
        index_list.reverse()

        sample_prediction = []
        sample_scores = []
        sample_links = []
        for index in index_list:
            e_id = entity_list[index]
            e_title = id2title[e_id]
            e_url = id2url[e_id]
            sample_prediction.append(e_title)
            sample_scores.append(scores_list[index])
            sample_links.append(e_url)
        predictions.append(sample_prediction)
        scores.append(sample_scores)
        links.append(sample_links)

    crossencoder_normalized_accuracy = -1
    overall_unormalized_accuracy = -1
    if not keep_all:
        crossencoder_normalized_accuracy = accuracy
        print(
            "crossencoder normalized accuracy: %.4f"
            % crossencoder_normalized_accuracy
        )

        if len(samples) > 0:
            overall_unormalized_accuracy = (
                crossencoder_normalized_accuracy *
                len(label_input) / len(samples)
            )
        print(
            "overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy
        )
    return (
        biencoder_accuracy,
        recall_at,
        crossencoder_normalized_accuracy,
        overall_unormalized_accuracy,
        len(samples),
        predictions,
        links,
        scores,
        embeddings,
    )


if __name__ == '__main__':

    print(torch.cuda.is_available())

    logger = utils.get_logger(args.logging.output_path)

    models = load_models(args, logger)
    # elastic = _load_wiki(args,logger)
    data = [
        {
            "id": 0,
            "label": "unknown",
            "label_id": -1,
            "context_left": "".lower(),
            "mention": "Shakespeare".lower(),
            "context_right": "'s account of the Roman general Julius Caesar's murder by his friend Brutus is a meditation on duty.".lower(),
        },
        {
            "id": 1,
            "label": "unknown",
            "label_id": -1,
            "context_left": "Shakespeare's account of the Roman general".lower(),
            "mention": "Julius Caesar".lower(),
            "context_right": "'s murder by his friend Brutus is a meditation on duty.".lower(),
        }
    ]

    (biencoder_accuracy,
     recall_at,
     crossencoder_normalized_accuracy,
     overall_unormalized_accuracy,
     samples_length,
     predictions,
     scores,
     ) = run(args, logger, *models, data)

    print(predictions)
    print(scores)
