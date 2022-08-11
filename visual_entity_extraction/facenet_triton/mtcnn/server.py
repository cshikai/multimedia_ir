from flask import Flask, jsonify, request, Response
# from flask_cors import CORS, cross_origin
import io
import base64
from model.mtcnn import MTCNN
from PIL import Image



app = Flask(__name__)

mtcnn = MTCNN(image_size=160, margin=0, device='cuda:0', keep_all=True)

@app.route('/crop', methods =['POST'])
# @cross_origin()
def build():

    res_dict = {}
    im_b64 = request.json['image']
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    image = Image.open(io.BytesIO(img_bytes))
    # Only box and probability is returned; cropped img is omitted to reduce overhead
    # img can be cropped and resized using the box returned on client's end
    img_tensor, prob, box, = mtcnn(image, return_prob=True)
    if prob[0] == None:
        res_dict['img'] = []
        res_dict['prob']=[]
        res_dict['box']=[]
    else:
        res_dict['img']=img_tensor.tolist()
        res_dict['prob']=prob.tolist()
        res_dict['box']=box.tolist()
    # res_bytes = pickle.dumps(res_dict)
    # return None
    # return Response(res_bytes)
    return jsonify(res_dict)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)