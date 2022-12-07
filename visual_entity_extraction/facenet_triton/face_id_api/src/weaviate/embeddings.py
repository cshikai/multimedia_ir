import numpy as np
import os
import torch
import weaviate

from mtcnn.manager import MTCNNManager
from triton.manager import TritonManager


class Generator():

    def __init__(self):
        self.mtcnn = MTCNNManager()
        self.inference = TritonManager()

    def generate_embedding_from_b64(self, id, img_dict_list):
        """
        Generates the average embedding of the faces from a list of b64-converted images
        and saves it to emb_path

        INPUT:
        ------------------------------------
        id (int)            : A unique id for each human entity
        img_dict_list (list): A list of {"images":<b64-encoded image>} dict

        """
        # folder_path = os.path.join(folder_path, '') # Assert folders string to the right format
        avg_emb = torch.zeros(512)  # Reset Avg Embedding
        img_count = 0  # Reset Image Count
        for i in img_dict_list:
            face_data = self.mtcnn.crop_faces_from_b64(i)
            if len(face_data['img']) != 0:
                img_count += 1
                max_val = np.argmax(face_data['prob'])
                top_img = [face_data['img'][max_val]]
                img_embedding = self.inference.infer_with_triton(top_img)
                img_embedding = torch.from_numpy(img_embedding)
                avg_emb = avg_emb.add(img_embedding[max_val])
        if img_count == 0:
            return []
        avg_emb = avg_emb.div(img_count)
        return avg_emb


class Uploader():

    def __init__(self, weaviate_endpt, index):
        self.weaviate = weaviate.Client(weaviate_endpt)
        self.index = index  # Weaviate's equivalent of ES's "Index" is "Class"

    def save_emb(self, id, emb):
        data_obj = {
            "id_no": "{}".format(str(id))
        }
        self.weaviate.data_object.create(
            data_obj,
            self.index,
            vector=emb
        )


class Identify():

    def __init__(self, weaviate_endpt):
        self.weaviate = weaviate.Client(weaviate_endpt)
