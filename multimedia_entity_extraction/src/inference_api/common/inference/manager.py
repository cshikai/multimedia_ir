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

        self.output = {}
        self.num_batch = 0

        data_generator = self.reader.get_generator(indexes)
        data_count = 0
        for data in data_generator:
            if data_count == 0:
                batch_data = {}
            data_count += 1
            processed_data_slice = self.processor.preprocess_for_triton(
                **data)
            for key, value in processed_data_slice.items():
                batch_data[key] = batch_data.get(key, []) + [value]

            if data_count == self.triton_manager.triton_cfg['max_batch']:
                self._infer_single_batch(batch_data)
                data_count = 0

        if data_count:
            self._infer_single_batch(batch_data)

        if self.output:
            merge_type_list, text_entity_id_list, visual_entity_id_list = self.writer.write(
                **self.output)
        else:
            merge_type_list, text_entity_id_list, visual_entity_id_list = [], [], []
        return merge_type_list, text_entity_id_list, visual_entity_id_list

    def _infer_single_batch(self, batch_data):
        '''
        Carry out inference for a batch that is at most as big as the batch size specific by the triton model
        '''

        self.num_batch += 1
        self.logger.info(
            '{} - Begin Batch {}'.format(datetime.now(), self.num_batch))

        batch_input_data, metadata = self.processor.collate_for_triton(
            **batch_data)

        # try:
        batch_output_data = self.triton_manager.infer_with_triton(
            batch_input_data)
        # except:
        #     self.logger.info(
        #         '{} - Unable to get results from Triton Server'.format(datetime.now()))
        #     raise ConnectionError('Unable to get response from Triton Server')

        batch_postprocessed_output = self.processor.postprocess_from_triton(
            batch_output_data, metadata)

        for key, value in batch_postprocessed_output.items():
            self.output[key] = self.output.get(key, []) + value
