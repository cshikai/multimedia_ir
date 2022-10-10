

import requests
import numpy as np
from spacy.lang.en import English
import os
import io
from PIL import Image
import base64

import pandas as pd
from sqlalchemy import Table, MetaData, create_engine
import dask.dataframe as dd
from elasticsearch import Elasticsearch

from inference_api.common.inference.data_reader import DataReader


class VALiveDataReader(DataReader):

    ELASTIC_URL = "https://elasticsearch:9200"

    def __init__(self):
        self.text_entity_extractor = UnknownTextEntityExtractor()
        self.visual_entity_extractor = UnknownVisualEntityExtractor()
        self.client = Elasticsearch(self.ELASTIC_URL,
                                    basic_auth=('elastic', 'changeme'),
                                    verify_certs=False
                                    )

    def get_generator(self, indexes):
        index = 0
        for document_id in indexes:
            result = self.client.get(index='documents_m2e2',
                                     id=document_id,
                                     )

            visual_entities = sorted(
                result['_source']['visual_entities'], key=lambda x: x['file_name'].split('/')[-1])
            text_caption_entities = sorted(
                result['_source']['text_caption_entities'], key=lambda x: x['file_name'].split('/')[-1])
            text_content = result['_source']['image_captions']

            # For each image in the article
            for img_ent, text_ent, caption in zip(visual_entities, text_caption_entities, text_content):
                image_generator = self.visual_entity_extractor.get_generator([
                                                                             img_ent])
                text_generator = self.text_entity_extractor.get_generator(
                    caption, text_ent)
                for image_url, image_entity_index, image_data, bounding_box, object_type, linked_image in image_generator:  # For each visual entity
                    for text, text_entity_index, token_span, linked_text in text_generator:  # For each textual entity
                        if not (linked_image and linked_text):
                            yield {
                                'index': document_id,
                                'image_url': image_url,
                                'text': text,
                                'text_entity_index': text_entity_index,
                                'image_entity_index': image_entity_index,
                                'image': image_data,
                                'token_span': token_span,
                                'bounding_box': bounding_box,
                                'object_type': object_type}


class UnknownVisualEntityExtractor:

    def __init__(self):
        pass

    def get_generator(self, images):

        for image in images:

            object_type = 'face'
            entity_ids = image['person_id']
            bounding_boxes = image['person_bbox']
            image_url = image['file_name']

            image_data = self.download_image_b64(image_url)

            img_bytes = base64.b64decode(image_data.encode('utf-8'))
            PIL_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            N = len(entity_ids)

            for i in range(N):
                entity_id = entity_ids[i]
                if entity_id == "-1":
                    linked_image = False
                else:
                    linked_image = True

                bounding_box = bounding_boxes[i]
                reshaped_bounding_boxes = [
                    (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3])]
                yield image_url, i, PIL_image, reshaped_bounding_boxes, object_type, linked_image

            object_bounding_boxes = image['obj_bbox']
            object_classes = image['obj_class']
            object_confidence = image['obj_conf']
            linked_image = False
            N = len(object_bounding_boxes)
            for i in range(N):
                object_class = object_classes[i]
                bounding_box = object_bounding_boxes[i]

                reshaped_bounding_boxes = [
                    (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3])]
                yield image_url, i, PIL_image, reshaped_bounding_boxes, object_class, linked_image

    def download_image(self, server_path):
        body = {'server_path': server_path}
        r = requests.get('http://image_server:8000/download/', json=body)
        image = np.asarray(r.json()['image'])
        return image

    def download_image_b64(self, server_path):
        body = {'server_path': server_path}
        r = requests.get('http://image_server:8000/download/', json=body)
        return r.json()['image']


class UnknownTextEntityExtractor:

    def __init__(self):
        self.token_mapper = TextTokenMapper()

    def get_generator(self, text, text_entities):
        N = len(text_entities['entity_links'])
        sentences = self.token_mapper.split_sentences(text)
        # print("Sentence:", sentences)

        for i in range(N):
            #             entity = text_entities[i]
            if text_entities['entity_links'][i] == "-1":
                linked_text = False
            else:
                linked_text = True
            sentence_index, span_start, span_end = text_entities['mention_spans'][i]
            sentence = sentences[sentence_index]
            # print("SENTENCE:", sentence, span_start, span_end)
            token_span = self.token_mapper.get_tokens(
                sentence, text_entities['mentions'][i], span_start, span_end)
            yield sentences[sentence_index], i, token_span, linked_text


class TextTokenMapper:

    def __init__(self):
        self.sentencizer = English()
        self.sentencizer.add_pipe('sentencizer')

    def split_sentences(self, text):
        paras = text.split("\n")
        sentences = []
        for para in paras:
            str_sents = list(self.sentencizer(para).sents)
            for sent in str_sents:
                tokens = list(self.sentencizer.tokenizer(sent.text))
                tokens = [token.text for token in tokens]
                if len(tokens) > 0:
                    sentences.append(sent.text)
        return sentences

    def get_tokens(self, sentence, entity_mention, span_start, span_end):

        # tokens = re.split('\W+', sentence)
        # print(sentence[span_start:span_end])
        # print(sentence)
        tokens = sentence.split(" ")
        # print(tokens)
        char_map = {}
        char_index = 0
        for index, token in enumerate(tokens):
            char_map[char_index] = index
            char_index += len(token)+1
        # print(char_map)
        start_token = char_map[span_start]

        end_token = len(tokens)  # If entity is last token in the sentence
        for i in char_map:
            if i < span_end:
                continue
            else:
                end_token = char_map[i]
                break

        # print(' '.join(tokens[start_token:end_token]))
        return (start_token, end_token)

    # def get_tokens(self, sentence, entity_mention, span_start, span_end):
    #     tokens = sentence.split(' ')
    #     # tokens = re.split('\W+', sentence)
    #     print(tokens)
    #     N = len(tokens)
    #     entity_tokens = len(entity_mention.split(' '))
    #     # entity_tokens = len(re.split('\W+', entity_mention))
    #     cumilative_char_index = 0
    #     char_to_token_span = []

    #     for token_index, token in enumerate(tokens):
    #         end_char_index = cumilative_char_index + len(token)
    #         char_to_token_span.append(
    #             (cumilative_char_index, end_char_index, token_index))
    #         cumilative_char_index = end_char_index + 1
    #     print(char_to_token_span)
    #     left = 0
    #     char_2_token_spans = {}
    #     print("START, STOP", entity_tokens-1, N)
    #     for right in range(entity_tokens-1, N):
    #         left_start_index, left_end_index, left_token_index = char_to_token_span[left]
    #         right_start_index, right_end_index, right_token_index = char_to_token_span[right]
    #         char_2_token_spans[(left_start_index, right_end_index)] = (
    #             left_token_index, right_token_index)
    #         left += 1

    #     print(sentence[span_start:span_end])
    #     while span_start > 0 and sentence[span_start - 1] != ' ':
    #         span_start -= 1
    #     while span_end < len(sentence) and sentence[span_end] != ' ':
    #         print(sentence[span_end])
    #         span_end += 1
    #     print(sentence[span_start:span_end])
    #     print(char_2_token_spans)
    #     token_span = char_2_token_spans[(span_start, span_end)]

    #     return token_span


class VADataReader(DataReader):

    DATA_ROOT = '/data/'

    def __init__(self):
        self.root_folder = os.path.join(self.DATA_ROOT, 'valid', 'manifest')
        self.data = dd.read_parquet(os.path.join(self.root_folder, 'data.parquet'),
                                    columns=['filename', 'caption'],
                                    engine='fastparquet')

    def get_generator(self, indexes):
        for index in indexes:
            yield self.read(index)

    def read(self, index):
        data_slice = self.data.loc[index].compute()
        text = self.read_text(data_slice)
        image, image_url = self.read_image(data_slice)

        text_entity_index = 1
        image_entity_index = 1
        token_span = (1, 1)
        bounding_box = [(1, 4), (3, 2)]

        return {
            'index': index,
            'image_url': image_url,
            'text': text,
            'image': image,
            'text_entity_index': text_entity_index,
            'image_entity_index': image_entity_index,
            'token_span': token_span,
            'bounding_box': bounding_box
        }

    def read_text(self, data_slice):
        text = data_slice['caption'].values[0]
        return text

    def read_image(self, data_slice):
        image_url = data_slice['filename'].values[0]
        image = Image.open(os.path.join(
            self.DATA_ROOT,  image_url))

        # convert greyscale to rgb
        if len(image.split()) != 3:
            rgbimg = Image.new("RGB", image.size)
            rgbimg.paste(image)
            image = rgbimg

        return image, image_url
