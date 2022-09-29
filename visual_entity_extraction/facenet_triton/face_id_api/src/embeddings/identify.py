import numpy as np
import torch
import yaml

from torch.nn.modules.distance import PairwiseDistance
from mtcnn.manager import MTCNNManager
from triton.manager import TritonManager
from elasticsearch import Elasticsearch


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()
ELASTIC_URL = config['ELASTICSEARCH']['URL']
INDEX_NAME = config['ELASTICSEARCH']['INDEX_NAME']
ELASTIC_USERNAME = config['ELASTICSEARCH']['ELASTIC_USERNAME']
ELASTIC_PASSWORD = config['ELASTICSEARCH']['ELASTIC_PASSWORD']


class Identify():

    def __init__(self, local):
        self.l2 = PairwiseDistance(p=2)
        self.mtcnn = MTCNNManager()
        self.facenet = TritonManager()
        self.local = local

        if not self.local:
            self.client = Elasticsearch(ELASTIC_URL,
                                        # ca_certs="",
                                        verify_certs=False,
                                        basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))

    # def refresh_embedding(self, id_list, embeddings):
    #     self.id_list = id_list
    #     self.embeddings = embeddings
    #     self.norm_embeddings = torch.stack([x/torch.linalg.norm(x) for x in self.embeddings]) # Used for cosine similarity
    #     self.norm_embeddings = torch.transpose(self.norm_embeddings,0,1)

    def _id_by_cosine(self, source_emb, norm_embeddings, id_list, k=1):
        """
        Performs dot product of source emb [1, 512] and transposed emb matrix [512, N]
        to obtain cosine similarity of each id. 

        INPUT:
        ------------------------------------
        source_emb          : Tensor of dim [1, 512]
        k                   : top k results to return

        """
        source_emb = source_emb/torch.linalg.norm(source_emb)
        cos_sim = torch.matmul(source_emb, norm_embeddings)
        topk_cos = torch.topk(cos_sim, k)
        conf = topk_cos.values.tolist()[0]
        topk_cos = topk_cos.indices.tolist()[0]
        results = [id_list[index] for index in topk_cos]

        return results, conf

    def _id_by_euc(self, source_emb, embeddings, id_list, k=1):
        """
        Computes L2 distance row by row on the matrix

        INPUT:
        ------------------------------------
        source_emb          : Tensor of dim [1, 512]
        k                   : top k results to return
        """
        euc_dist = []
        for entry in embeddings:
            euc_dist.append(self.l2(source_emb, entry).item())
        euc_dist = np.array(euc_dist)
        topk = np.argpartition(euc_dist, k)[:k].tolist()
        results = [id_list[index] for index in topk]
        conf = euc_dist[topk].tolist()
        return results, conf

    def _crop_and_get_emb(self, img_dict):
        res = self.mtcnn.crop_faces_from_b64(img_dict)
        if len(res['prob']) == 0:
            return None, None
        emb = self.facenet.infer_with_triton(res['img'])
        return res, emb

    def _id_by_cosine(self, source_emb, norm_embeddings, id_list, k=1):
        """
        Performs dot product of source emb [1, 512] and transposed emb matrix [512, N]
        to obtain cosine similarity of each id. 

        INPUT:
        ------------------------------------
        source_emb          : Tensor of dim [1, 512]
        k                   : top k results to return

        """
        source_emb = source_emb/torch.linalg.norm(source_emb)
        cos_sim = torch.matmul(source_emb, norm_embeddings)
        topk_cos = torch.topk(cos_sim, k)
        conf = topk_cos.values.tolist()[0]
        topk_cos = topk_cos.indices.tolist()[0]
        results = [id_list[index] for index in topk_cos]

        return results, conf

    def compare(self, img_dict, embeddings, norm_embeddings, id_list):
        res, embs = self._crop_and_get_emb(img_dict)
        if isinstance(embs, type(None)):
            return [], [], [], [], []
        if len(embeddings) == 0:
            return [], [], [], [], []
        embs = torch.from_numpy(embs)
        all_cos_id = []
        all_euc_id = []
        all_cos_conf = []
        all_euc_conf = []
        for i in embs:
            cur_emb = torch.unsqueeze(i, dim=0)
            face_id_cos, conf_cos = self._id_by_cosine(
                cur_emb, norm_embeddings, id_list, k=1)
            face_id_euc, conf_euc = self._id_by_euc(
                cur_emb, embeddings, id_list, k=1)
            all_cos_id.append(face_id_cos)
            all_cos_conf.append(conf_cos)
            all_euc_id.append(face_id_euc)
            all_euc_conf.append(conf_euc)
        bbox_int = [[int(x) for x in y] for y in res['box']]
        return all_cos_id, all_cos_conf, all_euc_id, all_euc_conf, bbox_int

    def compare_es(self, img_dict):
        res, embs = self._crop_and_get_emb(img_dict)
        if isinstance(embs, type(None)):
            return [], [], [], [], [], []
        ori_embs = embs
        embs = torch.from_numpy(embs)
        all_cos_id = []
        all_euc_id = []
        all_cos_conf = []
        all_euc_conf = []
        for emb in embs:
            cosine_search_query = {
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "filter": {
                                    "exists": {
                                        "field": "face_emb"
                                    }
                                }
                            }
                        },
                        "script": {
                            "source": "cosineSimilarity(params.queryVector, 'face_emb') + 1.0",
                            "params": {
                                "queryVector": emb.tolist()
                            }
                        }
                    }
                }
            }
            cosine_results = self.client.search(
                body=cosine_search_query, index=INDEX_NAME)

            """
            Unlike cosineSimilarity that represent similarity, 
            l1norm and l2norm shown below represent distances or differences. 
            This means, that the more similar the vectors are, 
            the lower the scores will be that are produced by the l1norm and l2norm functions. 
            Thus, as we need more similar vectors to score higher, 
            we reversed the output from l1norm and l2norm. 
            Also, to avoid division by 0 when a document vector matches the query exactly, 
            we added 1 in the denominator.
            """

            euc_search_query = {
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "filter": {
                                    "exists": {
                                        "field": "face_emb"
                                    }
                                }
                            }
                        },
                        "script": {
                            "source": "1 / (1 + l2norm(params.queryVector, 'face_emb'))",
                            "params": {
                                "queryVector": emb.tolist()
                            }
                        }
                    }
                }
            }
            euc_results = self.client.search(
                body=euc_search_query, index=INDEX_NAME)

            all_cos_id.append(cosine_results['hits']['hits'][0]['_id'])
            # Elasticsearch doesn't allow negative score, hence in ES query, there is a +1, making the range 0-2
            # Over here, we subtract 1 to compensate for the add-one. Cos Sim of 1 = most similar, -1 = most dissimilar
            all_cos_conf.append(cosine_results['hits']['hits'][0]['_score']-1)
            all_euc_id.append(euc_results['hits']['hits'][0]['_id'])
            all_euc_conf.append(euc_results['hits']['hits'][0]['_score'])
        bbox_int = [[int(x) for x in y] for y in res['box']]
        return all_cos_id, all_cos_conf, all_euc_id, all_euc_conf, bbox_int, ori_embs.tolist()
