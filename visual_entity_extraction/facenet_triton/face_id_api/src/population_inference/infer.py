
import base64
import json
import requests
import yaml

from PIL import Image, ImageDraw


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()

if __name__ == '__main__':

    img_file = config['infer']['img_file']

    with open(img_file, "rb") as f:
        im_bytes = f.read()
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    payload = json.dumps({"image": im_b64})

    r = requests.post(
        '{}/infer'.format(config['endpt']['fn_endpt']), data=payload, headers=headers)
    res = json.loads(r.text)
    print(res)

    with Image.open(img_file) as im:
        for i, j in enumerate(res['cos_id']):
            draw = ImageDraw.Draw(im)
            draw.rectangle(res['bb'][i])
            draw.text((res['bb'][i][0], res['bb'][i][1]), "id:"+str(j) +
                      " conf:"+str(res['cos_conf'][i]))  # Top left corner
        im.save(img_file[:-4]+"_inferred"+img_file[-4:])
