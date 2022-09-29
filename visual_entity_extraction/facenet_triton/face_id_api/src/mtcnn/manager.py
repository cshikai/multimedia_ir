import requests
import base64
import json
import yaml


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()
URL = config['MTCNN']['URL']


class MTCNNManager():

    def __init__(self):
        self.path = URL+'/crop'
        self.header = {'Content-type': 'application/json',
                       'Accept': 'text/plain'}

    def crop_faces_from_path(self, img_path):
        """        
        INPUT:
        ------------------------------------
        img_path:   Path to .jpg file

        RETURNS:
        ------------------------------------
        res:    Result dict comprising of:
                - res['img']:   normalised image in the form of NCHW or NCWH i dont really know lol
                                example shape: [N, 3, 224, 224]
                - res['box']:   bounding box of faces
                                example shape: [N, 4]
                - res['prob']:  confidence score of each face cropped
                                example shape: [N, 1]
                where N is the number of faces found
        """
        with open(img_path, "rb") as f:
            im_bytes = f.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        payload = json.dumps({"image": im_b64})
        r = requests.post(self.path, data=payload, headers=self.header)
        res = json.loads(r.text)
        return res

    def crop_faces_from_b64(self, img_bytes):
        """        
        This method is for FastAPI to directly pass the dict forward to MTCNN, instead of reading and re-wrapping the image. 
        INPUT:
        ------------------------------------
        img_bytes:  Image dict in the format of:
                - img_bytes['image']:   Original image in b64 format

        RETURNS:
        ------------------------------------
        res:    Result dict comprising of:
                - res['img']:   normalised image in the form of NCHW or NCWH i dont really know lol
                                example shape: [N, 3, 224, 224]
                - res['box']:   bounding box of faces
                                example shape: [N, 4]
                - res['prob']:  confidence score of each face cropped
                                example shape: [N, 1]
                where N is the number of faces found
        """
        r = requests.post(self.path, data=json.dumps(
            img_bytes), headers=self.header)
        res = json.loads(r.text)
        return res
