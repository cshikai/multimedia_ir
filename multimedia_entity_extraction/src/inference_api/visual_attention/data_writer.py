from elasticsearch import Elasticsearch


from inference_api.common.inference.data_writer import DataWriter
from evaluation.word_attention_generator import WordHeatmapGenerator
from evaluation.bounding_box_generator import BoundingBoxGenerator

class VADataWriter(DataWriter):

    ELASTIC_URL = "http://elasticsearch:9200"

    def __init__(self):
        super().__init__()
        self.heatmap_generator = WordHeatmapGenerator()
        # self.bounding_box_generator = BoundingBoxGenerator()
        self.client = Elasticsearch(self.ELASTIC_URL)
    def write(self, **kwargs):

        unknown_text_entites = kwargs['unknown_text_entities']

        unknown_visual_entities = kwargs['unknown_visual_entities']

        num_text = len(unknown_text_entites)
        num_visual = len(unknown_visual_entities)
        
        text_entity_ids = [i for i in range(num_text)]

        visual_entity_ids  = [i for i in range(num_text,num_text+num_visual)]

        link_id = num_text+num_visual
        for i,text_entity in enumerate(unknown_text_entites):
            for j,visual_entity in enumerate(unknown_visual_entities):
                entity_linked = self.link_entities()
                if entity_linked:
                    text_entity_ids[i] = link_id
                    visual_entity_ids[j] = link_id
                    link_id += 1
        
        #client.update(index='documents',id=document_id,doc=new_doc)
        # self.bounding_box_generator.generate(**kwargs)
        return {}

    def _get_global_entity_id(self,text_entity_ids,visual_entity_ids):
        
    def link_entities(self):
