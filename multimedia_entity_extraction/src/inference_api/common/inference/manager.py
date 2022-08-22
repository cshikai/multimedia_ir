import os
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from triton.manager import TritonManager
from .config import cfg


class InferenceManager(ABC):

    def __init__(self, log_db_manager, triton_cfg):
        self.log_db_manager = log_db_manager
        self.type_string = self.log_db_manager.type_string

        log_dir = os.path.join(
            cfg.inference_manager.base_log_directory, self.type_string, 'log.log')

        if not os.path.exists(os.path.dirname(log_dir)):
            os.makedirs(os.path.dirname(log_dir))

        logging.basicConfig(filename=log_dir)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.triton_manager = TritonManager(triton_cfg)

    writer = NotImplemented
    reader = NotImplemented
    processor = NotImplemented

    def infer(self, indexes):
        '''
        Carry out inference for multi data sample

        indexes : list of integers that identifies the data sample

        '''

        output = {}
        index_len = len(indexes)

        num_batch = (index_len//self.triton_manager.triton_cfg['max_batch']) + \
            (index_len % self.triton_manager.triton_cfg['max_batch'] > 0)

        for b in range(num_batch):
            self.logger.info(
                '{} - Begin Batch {} out of {} '.format(datetime.now(), b+1, num_batch))
            start_index = b*self.triton_manager.triton_cfg['max_batch']
            end_index = (b+1)*self.triton_manager.triton_cfg['max_batch']
            batch_output = self._infer_single_batch(
                indexes[start_index:end_index])
            for key, value in batch_output.items():
                output[key] = output.get(key,[]) + [value]
        return output

    def _infer_single_batch(self, indexes):
        '''
        Carry out inference for a batch that is at most as big as the batch size specific by the triton model
        '''

        batch_data = {}

        for index in indexes:
            index_data = self.reader.read(index)
            processed_data_slice = self.processor.preprocess_for_triton(
                **index_data)
            for key, value in processed_data_slice.items():
                batch_data[key] = batch_data.get(key,[]) + [value]

        batch_input_data, metadata = self.processor.collate_for_triton(**batch_data)

        # try:
        output_data = self.triton_manager.infer_with_triton(batch_input_data)
        # except:
        #     self.logger.info(
        #         '{} - Unable to get results from Triton Server'.format(datetime.now()))
        #     raise ConnectionError('Unable to get response from Triton Server')

        readable_results = self.processor.postprocess_from_triton(output_data,metadata)

        output = self.writer.write(**readable_results)

        return output