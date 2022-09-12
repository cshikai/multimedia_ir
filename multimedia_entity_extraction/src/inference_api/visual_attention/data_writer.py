from elasticsearch import Elasticsearch


from inference_api.common.inference.data_writer import DataWriter
from evaluation.word_attention_generator import WordHeatmapGenerator
from evaluation.bounding_box_generator import BoundingBoxGenerator
from evaluation.box_heatmap_aggregator import BoxHeatmapAggregator


from heapq import *

class VADataWriter(DataWriter):

    ELASTIC_URL = "http://elasticsearch:9200"

    def __init__(self):
        super().__init__()
        # self.heatmap_generator = WordHeatmapGenerator()
        self.box_aggregator = BoxHeatmapAggregator()
        # self.bounding_box_generator = BoundingBoxGenerator()
        self.client = Elasticsearch(self.ELASTIC_URL,
                                    basic_auth=('elastic', 'changeme'),
                                    verify_certs=False
                                    )
        self.HEATMAP_THRESHOLD = 1
    def write(self, **kwargs):

        max_heap = []
        aggregated_heatmap_scores = self.box_aggregator.aggregate(**kwargs)

        text_entities_index = kwargs['text_entity_index']
        visual_entities_index = kwargs['image_entity_index']
        document_ids = kwargs['indexes']

        N = len(aggregated_heatmap_scores)
        text_entities_set = set(text_entities_index)
        visual_entities_set = set(visual_entities_index)

        for i in range(N):
            heap_item = (
                -aggregated_heatmap_scores[i],
                visual_entities_index[i],
                text_entities_index[i],
                document_ids[i]
                )
            heappush(max_heap,heap_item)
        
        print(max_heap)
        while max_heap:
            score,visual_entity_id,text_entity_id,document_id = heappop(max_heap)

            if -score > self.HEATMAP_THRESHOLD:
                if visual_entity_id in visual_entities_set and text_entity_id in text_entities_set:
                    text_entities_set.remove(text_entity_id)
                    visual_entities_set.remove(visual_entity_id)
                    new_entity_id = '_'.join([str(document_id),str(text_entity_id),str(visual_entity_id)])
                    self.update_elastic(document_id,text_entity_id,visual_entity_id,new_entity_id)
            else:
                break

    def update_elastic(self,document_id, text_entity_id, visual_entity_id):

        self.client.update(index='documents',id=document_id,
                body={"doc":'es' })
        #client.update(index='documents',id=document_id,doc=new_doc)