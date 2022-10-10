
from datetime import datetime

from sqlalchemy import update, bindparam
import sqlite3

from inference_api.common.inference.manager import InferenceManager


from .data_reader import VALiveDataReader
from .data_writer import VADataWriter
from .data_process.processor import VAProcessor


from .triton_config.config import cfg as triton_cfg
from triton.dummy_triton_manager import DummyTritonManager


class VAManager(InferenceManager):

    def __init__(self, log_db_manager):

        super().__init__(log_db_manager, triton_cfg)

        self.triton_manager = DummyTritonManager()
        self.writer = VADataWriter()
        self.reader = VALiveDataReader()
        self.processor = VAProcessor()

    def get_heatmap(self, indexes):

        # primary_key = self.log_db_manager.register_job(batch_date=date)

        # if primary_key > -1:

        # text, image = self.reader.read(indexes)

        # self.logger.info('{} - Specific Date All Tracks Only : Classifying {} Tracks on Date {}'.format(
        #     datetime.now(), len(track_ids), date))

        # self.log_db_manager.update_job(primary_key, 1)

        self.infer(indexes)
        # self.log_db_manager.update_job(primary_key, 2)

        # except:
        #     self.log_db_manager.update_job(primary_key, 3)
        # else:
        #     self.logger.info('{} - Batch Classification Attempted on Date {} but was already executed, skipping.'.format(
        #         datetime.now(), date))

    # def classify_unclassed_tracks_on_latest_date(self):

    #     date = self.reader.get_last_unclassified_date()
    #     track_ids = self.reader.get_unclassified_tracks_by_date(date)

    #     self.logger.info('==== Mode - Latest Date Unclassed Only : Classifying {} Tracks on Date {}'.format(
    #         len(track_ids), date))

    #     self.infer(track_ids)

    # def classify_remainder_tracks_on_date(self, date):
    #     track_ids = self.reader.get_unclassified_tracks_by_date(date)

    #     self.logger.info('==== Mode - Specific Date Remainder Only : Classifying {} Tracks on Date {}'.format(
    #         len(track_ids), date))

    #     self.classify_tracks(track_ids, is_remainder=True)

    # def classify_all_unclassed_tracks(self, dates):
    #     for date in dates:
    #         self.classify_unclassed_tracks_on_date(date)
#####################################################################
