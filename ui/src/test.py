from app import search_reports, get_report, get_entity_name
import base64
from PIL import Image, ImageDraw, ImageFont
import io

reports = search_reports('*')

for report in reports:
    report = get_report(report.id)

    for server_path, image in report.images.items():
        img_bytes = base64.b64decode(image.encode('utf-8'))
        im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # im = Image.fromarray(np.asarray(image).astype(np.uint8))
        for visual_entity in report.visual_entities:
            if visual_entity["file_name"] != server_path:
                continue
            else:
                # Generate face_id checkbox
                for person_id in set(visual_entity['person_id']):
                    person_name = get_entity_name(person_id)

                # Generate face_id bounding box
                for person_idx, bbox in enumerate(visual_entity['person_bbox']):
                    draw = ImageDraw.Draw(im)
                    draw.rectangle(
                        bbox, outline='green')
                    # Top left corner
                    draw.text((bbox[0], bbox[1]),
                              f"{get_entity_name(visual_entity['person_id'][person_idx])}, Conf: {visual_entity['person_conf'][person_idx]}", font=ImageFont.truetype("DejaVuSans.ttf", 12))

                # Generate obj_det bounding box
                for obj_idx, bbox in enumerate(visual_entity['obj_bbox']):
                    draw = ImageDraw.Draw(im)
                    draw.rectangle(
                        bbox, outline='blue')
                    # Top left corner
                    draw.text((bbox[0], bbox[1]),
                              f"{visual_entity['obj_class'][obj_idx]}, Conf: {visual_entity['obj_conf'][obj_idx]}", font=ImageFont.truetype("DejaVuSans.ttf", 12))
                break
        captions = report.image_captions[server_path] if server_path in report.image_captions else ''
    print(report.id)
