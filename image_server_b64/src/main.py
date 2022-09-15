from http import server
import os
from typing import List

import h5py
import base64
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

api = FastAPI(
    title='Image Server API',
    description='API for Image Server. Standin for a S3 for prototyping purposes',
    version='1.0.0'
)


class ImageData(BaseModel):

    filename: str
    image: str


class ImageMetaData(BaseModel):

    server_path: str


@api.put("/upload/")
def upload_image(image_data: ImageData):
    image_data = image_data.dict()
    writer = M2E2ImageWriter()
    # writer = F1ImageWriter()
    response = {}
    try:
        response['server_path'] = writer.write_single_image(
            image_data['filename'], image_data['image'])
    except:
        response['server_path'] = 'upload_failure'

    return response


@api.get("/download/")
def download_image(image_metadata: ImageMetaData):

    server_path = image_metadata.dict()['server_path']
    reader = F1ImageReader()

    response = {}
    try:

        response['image'] = reader.read_single_image(server_path)
        response['status'] = 'success'
    except:
        response['image'] = None
        response['status'] = 'failure'
    return response


class M2E2ImageWriter:
    def __init__(self):

        self.image_root = '/images'
        self.prefix = 'M2E2'

    def _b64_reader():
        print("Do something")

    def _b64_writer():
        print("Do something")

    def write_single_image(self, filename, image):
        print(filename)
        article_id, image_id = filename.rsplit('_', 1)
        folder_path = os.path.join(self.image_root, self.prefix, article_id)
        file_path = os.path.join(
            self.image_root, self.prefix, article_id, image_id + ".h5")

        if os.path.exists(file_path):
            return file_path
        else:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file = h5py.File(file_path, "w")
            dt = h5py.special_dtype(vlen=str)
            _dataset = file.create_dataset(
                "image", len(image), dtype=dt)
            _dataset.attrs["img"] = image

            file.close()
            return file_path


class F1ImageWriter:
    def __init__(self):

        self.image_root = '/images'
        self.prefix = 'f1'

    def write_single_image(self, filename, image):

        article_id, image_id = filename.split('_')
        folder_path = os.path.join(self.image_root, self.prefix, article_id)
        file_path = os.path.join(
            self.image_root, self.prefix, article_id, image_id + ".h5")

        if os.path.exists(file_path):
            return file_path
        else:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file = h5py.File(file_path, "w")
            _dataset = file.create_dataset(
                "image", np.shape(image), h5py.h5t.STD_U8BE, data=image)

            file.close()
            return file_path


class F1ImageReader:

    # def __init__(self):
    #     self.image_root = '/images'
    #     self.prefix = 'f1'

    def read_single_image(self, filepath):

        with h5py.File(filepath, "r") as f:

            image = f['image'].attrs['img']

        return image
