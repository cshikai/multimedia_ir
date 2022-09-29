import numpy as np
import os
import torch

from mtcnn.manager import MTCNNManager
from triton.manager import TritonManager
from embeddings.identify import Identify


class EmbeddingDataset():

    def __init__(self, emb_path, local):
        self.mtcnn = MTCNNManager()
        self.inference = TritonManager()
        self.id_module = Identify(local=local)
        self.local = local

        if self.local:
            self.emb_path = os.path.join(emb_path, '')
            self.id_list, self.embeddings = self._load_embeddings_local()

            if len(self.id_list) > 0:  # If not empty
                self.norm_embeddings = self._norm_transpose_emb(
                    self.embeddings)  # Used for cosine similarity
            else:
                self.norm_embeddings = []

    def _norm_transpose_emb(self, emb):
        norm_emb = torch.stack([x/torch.linalg.norm(x) for x in emb])
        norm_emb = torch.transpose(norm_emb, 0, 1)
        return norm_emb

    def generate_embedding(self, id, folder_path):
        """
        Generates the average embedding of the faces in a folder and saves it to emb_path

        INPUT:
        ------------------------------------
        id (int)            : A unique id for each human entity
        folder_path (str)   : Path to folder containing the images

        """
        folder_path = os.path.join(
            folder_path, '')  # Assert folders string to the right format
        avg_emb = torch.zeros(512)  # Reset Avg Embedding
        img_count = 0  # Reset Image Count

        for i in os.listdir(folder_path):
            face_data = self.mtcnn.crop_faces(folder_path+i)
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
        # Append new embedding to the list of existing embeddings
        usqz_emb = torch.unsqueeze(avg_emb, 0)
        if self.embeddings == []:  # empty list
            self.embeddings = usqz_emb
        else:
            self.embeddings = torch.cat([self.embeddings, usqz_emb])
        self.norm_embeddings = self._norm_transpose_emb(self.embeddings)
        self.id_list.append(id)
        return avg_emb

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

        if self.local:
            # Append new embedding to the list of existing embeddings
            usqz_emb = torch.unsqueeze(avg_emb, 0)
            if self.embeddings == []:  # empty list
                self.embeddings = usqz_emb
            else:
                self.embeddings = torch.cat([self.embeddings, usqz_emb])
            self.norm_embeddings = self._norm_transpose_emb(self.embeddings)
            self.id_list.append(id)

        return avg_emb

    def _load_embeddings_local(self):
        """
        Load all pre-generated embeddings from emb_path folder as a [N, 512] matrix. 

        RETURNS:
        ------------------------------------
        emb_list:   matrix of embeddings stacked together
                    example shape:  [N, 512]
        """
        # Populate list of ids
        id_list = []
        emb_files = os.listdir(self.emb_path)
        if len(emb_files) == 0:
            return [], []

        for i in emb_files:
            id_val = int(i[:-3])
            id_list.append(id_val)
        id_list.sort()
        # Populate embeddings
        emb_list = []
        for i in id_list:
            temp_emb = torch.load(self.emb_path+str(i)+'.pt')
            emb_list.append(temp_emb)

        emb_list = torch.stack(emb_list)
        # print(len(id_list))
        return id_list, emb_list

    def find_id(self, img_dict):
        if self.local:
            return self.id_module.compare(img_dict, self.embeddings, self.norm_embeddings, self.id_list)
        else:
            return self.id_module.compare_es(img_dict)
