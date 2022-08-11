import numpy as np
import tritonclient.http as httpclient
import yaml

def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

config = read_yaml()
URL = config['triton']['URL']
MODEL_NAME = config['triton']['model_name']

class TritonManager():

    VERBOSE = False

    def __init__(self):
        self.triton_client = httpclient.InferenceServerClient(
            url=URL, verbose=self.VERBOSE)

    def infer_with_triton(self, img):
        """
        Given that N is the number of faces found, 

        INPUT:
        ------------------------------------
        img:    Output from MTCNN (NCHW/NCWH in list format)
                example shape:  [N, 3, 160, 160]

        RETURNS:
        ------------------------------------
        emb:    embeddings of each cropped face
                example shape:  [N, 512]
        """
        img = np.array(img, dtype='float32')

        inputs = []
        outputs = []

        inputs.append(
            httpclient.InferInput(name="INPUT__0", shape=img.shape, datatype="FP32")
        )
        inputs[0].set_data_from_numpy(img, binary_data=False)
        
        outputs.append(httpclient.InferRequestedOutput(name="OUTPUT__0"))

        result = self.triton_client.infer(
            model_name=MODEL_NAME, 
            inputs=inputs, 
            outputs=outputs
        )

        result = result.as_numpy("OUTPUT__0")
        return result
