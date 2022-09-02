import numpy as np
import tritonclient.http as httpclient
import yaml

def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

config = read_yaml()
URL = config['triton']['URL']
MODEL_NAME = config['triton']['model_name']
BATCH_SZ = config['triton']['batch_size']

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

        result_list = []
        
        batches = int((len(img)-1)/BATCH_SZ)+1 # Min. of N is always 1, zero case handled by caller
        for batch in range(batches):
            inputs = []
            outputs = []
            data_batch = img[batch*BATCH_SZ:(batch+1)*BATCH_SZ]
            inputs.append(
                httpclient.InferInput(name="INPUT__0", shape=data_batch.shape, datatype="FP32")
            )
            inputs[0].set_data_from_numpy(data_batch, binary_data=False)
            
            outputs.append(httpclient.InferRequestedOutput(name="OUTPUT__0"))

            result = self.triton_client.infer(
                model_name=MODEL_NAME, 
                inputs=inputs, 
                outputs=outputs
            )
            result = result.as_numpy("OUTPUT__0")
            result_list.append(result)

        merged_result = np.concatenate(result_list,axis=0)
        return merged_result
