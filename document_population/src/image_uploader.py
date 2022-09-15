
import base64
from http import server
import requests
import os
from PIL import Image
import numpy as np


class F1ImageUploader:
    def __init__(self):
        self.server_address = 'image_server'
        self.image_root = '/images'
        self.prefix = 'm2e2'
        self.source_root = '/data/images_m2e2'

    def upload_single_image(self, filename):

        # full_name = os.path.join(self.source_root,filename+'.jpg')
        full_name = os.path.join(self.source_root, filename)
        print(full_name)

        image = np.asarray(Image.open(full_name).convert('RGB')).tolist()

        body = {
            'filename': filename.strip('.jpg'),
            'image': image
        }
        print("Body:", body['filename'], len(body['image']))
        # print(image)
        # body = {'filename':'a_0','image': [[1,2],[1,2]]}
        r = requests.put(
            'http://image_server:8000/upload/', json=body)

        return r.json()['server_path']
    
    def upload_single_image_b64(self, filename):

        # full_name = os.path.join(self.source_root,filename+'.jpg')
        full_name = os.path.join(self.source_root, filename)
        print(full_name)

        with open(full_name, "rb") as f:
            im_bytes = f.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")

        body = {
            'filename': filename.strip('.jpg'),
            'image': im_b64
        }
        print("Body:", body['filename'], len(body['image']))
        # print(image)
        # body = {'filename':'a_0','image': [[1,2],[1,2]]}
        r = requests.put(
            'http://image_server:8000/upload/', json=body)

        return r.json()['server_path']


# server_path = '/images/f1/1119/0.h5'
# print(server_path)

# body = {'server_path': server_path}
# r = requests.get(
#         'http://image_server:8000/download/', json=body)

# image = np.asarray(r.json()['image'])

# print(image.shape)
