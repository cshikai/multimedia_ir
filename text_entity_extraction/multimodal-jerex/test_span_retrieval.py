import requests
import json
from typing import List, Dict
import pandas as pd
import ast
import re
import time
import tqdm

from haystack.document_stores import ElasticsearchDocumentStore
from spacy.lang.en import English


def doc_sents(text):
    nlp = English()
    nlp.create_pipe('sentencizer')
    nlp.add_pipe('sentencizer')

    paras = text.split("\n")
    paras = [para for para in paras]

    sentences = []

    for doc_idx, para in enumerate(paras):

        str_sents = list(nlp(para).sents)

        num_tokens = 0
        for sent in str_sents:
            tokens = list(nlp.tokenizer(sent.text))
            tokens = [token.text for token in tokens]

            if len(tokens) > 0:
                num_tokens += len(tokens)
                sentence = {}
                sentence['tokens'] = tokens
                sentence['sentence'] = sent.text
                sentences.append(sentence)

    return sentences


if __name__ == '__main__':

    start = time.time()

    document_store = ElasticsearchDocumentStore(host="elasticsearch",
                                                port="9200",
                                                username="elastic",
                                                password="changeme",
                                                scheme="https",
                                                verify_certs=False,
                                                index='documents',
                                                search_fields=['content', 'title'])

    documents = document_store.get_all_documents()

    articles_df = pd.DataFrame(
        columns=['ID', 'title', 'text', 'elasticsearch_ID', 'text_entities'])

    for document in documents:
        articles_df.loc[-1] = [document.meta['ID'], document.meta['link'],
                               document.content, document.id, document.meta['text_entities']]  # adding a row
        articles_df.index = articles_df.index + 1  # shifting index
        articles_df = articles_df.sort_index()  # sorting by index

    print(articles_df.info())
    print(articles_df.head())

    articles_df['original_sents'] = articles_df['text'].apply(
        lambda text: doc_sents(text))

    for idx, row in articles_df.iterrows():
        text_entities = row['text_entities']
        if type(text_entities) == str:
            text_entities = ast.literal_eval(text_entities)

        mention_spans = [entity['sentence_char_span']
                         for entity in text_entities]

        mention = [entity['mention']
                   for entity in text_entities]

        sentences_dict_list = row['original_sents']
        sentences = [sentence['sentence'] for sentence in sentences_dict_list]
        for idx in range(len(mention_spans)):
            print("mention in ES: ",
                  mention[idx], " mention retrieved by span: ", sentences[mention_spans[idx][0]][mention_spans[idx][1]:mention_spans[idx][2]])
