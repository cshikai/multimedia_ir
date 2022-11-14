import json
import os
from copy import deepcopy
from tqdm import tqdm

# from inference_api.visual_attention_caption.manager import VAManager
# from inference_api.visual_attention_caption.log_manager import VALogDatabaseManager

from elasticsearch import Elasticsearch

ELASTIC_URL = "https://elasticsearch:9200"
client = Elasticsearch(ELASTIC_URL,
                       basic_auth=('elastic', 'changeme'),
                       verify_certs=False
                       )

# log_manager = VALogDatabaseManager('visual_attention_caption')
# va_manager = VAManager(log_manager)


def resolve_entity_name(entity_link):
    result = client.get(index='wikipedia',
                        id=entity_link,
                        )
    return result['_source']['title']


def rollback(ledger_obj):
    for article in ledger_obj.ledger:
        result = client.update(index='documents_m2e2',
                               id=article,
                               doc={"text_caption_entities": ledger_obj.ledger[article]['text_caption_entities'],
                                    "visual_entities": ledger_obj.ledger[article]['visual_entities']}
                               )
    # ledger_obj.clear_ledger()


def ingest(all_articles, ledger_obj):
    id_list = [x['_id'] for x in all_articles['hits']['hits']]
    for article_id in tqdm(id_list):
        print(article_id)
        merge_type_list, text_entity_id_list, visual_entity_id_list = va_manager.infer([
                                                                                       article_id])
        result = client.get(index='documents_m2e2',
                            id=article_id,
                            )
        existing_capt_dict = None
        existing_image_dict = None
        ledger_obj.add_entry(article_id, result['_source'])
        for vis, text, merge_type in zip(visual_entity_id_list, text_entity_id_list, merge_type_list):
            vis_data = vis.split('_')
            img_path = "_".join(vis_data[:-2])
            vis_type = vis_data[-2]
            vis_index = int(vis_data[-1])

            text_data = text.split('_')
            text_index = int(text_data[-1])

            existing_image_dict = result['_source']['visual_entities']
            existing_capt_dict = result['_source']['text_caption_entities']

            for index, text_caption in enumerate(existing_capt_dict):
                if text_caption['file_name'] == img_path:
                    caption_index = index
                    break

            for index, img_data in enumerate(existing_image_dict):
                if img_data['file_name'] == img_path:
                    img_index = index
                    break

            # For unknown entity, an id is generated; mention is used as entity name
            if merge_type == 'unk':
                resolved_id = "unk_" + \
                    "/".join(img_path.split("/")[-2:]) + \
                    '_{}'.format(str(vis_index))
                resolved_name = existing_capt_dict[caption_index]['mentions'][text_index]
                if vis_type == 'face':
                    existing_image_dict[img_index]['person_id'][vis_index] = resolved_id
                else:
                    existing_image_dict[img_index]['obj_class'][vis_index] = resolved_name
                existing_capt_dict[caption_index]['entity_links'][text_index] = resolved_id
                existing_capt_dict[caption_index]['entity_names'][text_index] = resolved_name

            # Unknown Image Entity, get resolved id from text
            elif merge_type == 'text':
                resolved_id = existing_capt_dict[caption_index]['entity_links'][text_index]
                # Prevent unk entity from being identified twice
                if resolved_id[:4] == 'unk_':
                    continue
                resolved_name = resolve_entity_name(resolved_id)
                # Write id for face, but entity name for object
                if vis_type == 'face':
                    existing_image_dict[img_index]['person_id'][vis_index] = resolved_id
                else:
                    existing_image_dict[img_index]['obj_class'][vis_index] = resolved_name

            # Unknown Text Entity, get resolved id from image
            elif merge_type == 'image':
                if vis_type == 'face':
                    resolved_id = existing_image_dict[img_index]['person_id'][vis_index]
                else:
                    resolved_id = existing_image_dict[img_index]['obj_class'][vis_index]
                # Prevent unk entity from being identified twice
                if resolved_id[:4] == 'unk_':
                    continue
                resolved_name = resolve_entity_name(resolved_id)
                existing_capt_dict[caption_index]['entity_links'][text_index] = resolved_id
                existing_capt_dict[caption_index]['entity_names'][text_index] = resolved_name

        # Update to ES
        if existing_capt_dict and existing_image_dict:
            result = client.update(index='documents_m2e2',
                                   id=article_id,
                                   doc={"text_caption_entities": existing_capt_dict,
                                        "visual_entities": existing_image_dict}
                                   )


class ledger():
    def __init__(self) -> None:
        if 'ledger.json' in os.listdir():
            with open('ledger.json', 'r') as fp:
                self.ledger = json.load(fp)
        else:
            self.ledger = {}

    def clear_ledger(self):
        self.ledger = {}
        with open('ledger.json', 'w') as fp:
            json.dump(self.ledger, fp)

    def add_entry(self, article_id, result_source):
        self.ledger[article_id] = {}
        self.ledger[article_id]['visual_entities'] = deepcopy(result_source)[
            'visual_entities']
        self.ledger[article_id]['text_caption_entities'] = deepcopy(result_source)[
            'text_caption_entities']
        with open('ledger.json', 'w') as fp:
            json.dump(self.ledger, fp)


if __name__ == "__main__":
    ledger_obj = ledger()
    all_articles = client.search(index='documents_m2e2',
                                 body={'size': 1000}
                                 )
    # ingest(all_articles, ledger_obj)
    rollback(ledger_obj)
