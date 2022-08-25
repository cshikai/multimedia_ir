import base64
import json
import os
import requests
import yaml


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()

img_folder = config['crawl']['save_path']

# To mount volume for dataset
if __name__ == '__main__':
    for person_id in os.listdir(img_folder):
        img_list = []
        for image in os.listdir(os.path.join(img_folder, person_id)):
            with open(f"{img_folder}/{person_id}/{image}", "rb") as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            img_list.append({'image': im_b64})
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        payload = json.dumps({"id": person_id, "images": img_list})

        r = requests.post(
            '{}/generate'.format(config['endpt']['fn_endpt']), data=payload, headers=headers)
