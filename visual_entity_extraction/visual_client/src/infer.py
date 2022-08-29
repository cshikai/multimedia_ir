
import base64
import json
import requests
import yaml
import os

from PIL import Image, ImageDraw


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()

if __name__ == '__main__':

    img_folder = config['infer']['img_folder']
    detection_dict = {}
    for img_file in os.listdir(img_folder):
        detection_dict[img_file] = {}
        with open(img_folder+'/'+img_file, "rb") as f:
            im_bytes = f.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")

        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        payload = json.dumps({"image": im_b64})

        r_fn = requests.post(
            '{}/infer'.format(config['endpt']['fn_endpt']), data=payload, headers=headers)
        res_fn = json.loads(r_fn.text)
        r_yolo = requests.post(
            '{}/infer'.format(config['endpt']['yolo_endpt']), data=payload, headers=headers)
        res_yolo = json.loads(r_yolo.text)
        

        detection_dict[img_file]['person_bbox'] = res_fn['bb']
        detection_dict[img_file]['person_id'] = res_fn['cos_id']
        detection_dict[img_file]['obj_bbox'] = [[int(element) for element in sublist[:4]]  for sublist in res_yolo['bboxes'][0]]
        detection_dict[img_file]['obj_class'] = [int(sublist[-1])  for sublist in res_yolo['bboxes'][0]]
        print(detection_dict)
        with Image.open(img_folder+'/'+img_file) as im:
            # Draw Facenet Bounding Boxes
            for i, j in enumerate(res_fn['cos_id']):
                draw = ImageDraw.Draw(im)
                draw.rectangle(res_fn['bb'][i])
                draw.text((res_fn['bb'][i][0], res_fn['bb'][i][1]), "id:"+str(j) +
                        " conf:"+str(res_fn['cos_conf'][i]))  # Top left corner
            for i, j in enumerate(detection_dict[img_file]['obj_bbox']):
                draw = ImageDraw.Draw(im)
                draw.rectangle(j)
                # draw.text((res_fn['bb'][i][0], res_fn['bb'][i][1]), "id:"+str(j) +
                #         " conf:"+str(res_fn['cos_conf'][i]))  # Top left corner
            im.save(img_file[:-4]+"_inferred"+img_file[-4:])
