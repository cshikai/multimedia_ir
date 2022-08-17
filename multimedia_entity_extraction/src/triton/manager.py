import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class TritonManager():

    URL = 'triton:8000'
    VERBOSE = False

    def __init__(self, triton_cfg):
        self.triton_client = httpclient.InferenceServerClient(
            url=self.URL, verbose=self.VERBOSE)
        self.headers = None
        self.triton_cfg = triton_cfg

    def infer_with_triton(self,
                          numpy_inputs
                          ):

        inputs = []
        outputs = []

        i = 0
        for key, value in numpy_inputs.items():
            shape = self.triton_cfg['inputs'][key]['shape']

            batch_size = value.shape[0]

            if shape[1] == -1:
                seq_len = value.shape[1]
                shape = [batch_size, seq_len] + \
                    self.triton_cfg['inputs'][key]['shape'][2:]

            else:
                shape = [batch_size, ] + \
                    self.triton_cfg['inputs'][key]['shape'][1:]

            inputs.append(httpclient.InferInput(
                key, shape, self.triton_cfg['inputs'][key]['type']))
            inputs[i].set_data_from_numpy(value, binary_data=False)
            i += 1

        for key, value in self.triton_cfg['outputs'].items():
            outputs.append(httpclient.InferRequestedOutput(
                key, binary_data=False))

        # optional
        query_params = None  # {'test_1': 1, 'test_2': 2}

        results = self.triton_client.infer(
            self.triton_cfg['model_name'],
            inputs,
            outputs=outputs,
            query_params=query_params,
            headers=self.headers,
        ).get_response()

        return results
