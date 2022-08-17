import os
import sqlite3
from abc import ABC, abstractmethod
from .config import cfg


class LogDatabaseManager(ABC):

    BASE_DIR = cfg.base_directory

    CREATE_JOB_DETAIL_TABLE_COMMAND: str

    ADD_JOB_DETAIL_TABLE_COMMAND: str

    # 'SELECT id FROM Jobs WHERE job_status!=3 AND job_date='

    LIMIT_RUN: bool

    CREATE_JOB_TABLE_COMMAND = '''
    CREATE TABLE IF NOT EXISTS Job(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_start TEXT NOT NULL,
        job_status INTEGER NOT NULL,
        job_last_update TEXT
    )
    '''

    CHECK_JOB_AVAILABILITY_COMMAND = 'SELECT id FROM Job WHERE job_status!=3 AND id in '

    ADD_JOB_COMMAND = '''
    INSERT INTO Job(job_start,job_status,job_last_update) VALUES(strftime('%Y-%m-%d %H:%M:%S','now'),0,strftime('%Y-%m-%d %H:%M:%S','now'))
    '''

    UPDATE_JOB_STATUS_COMMAND = '''UPDATE Job
    SET job_status = ?, job_last_update = strftime('%Y-%m-%d %H:%M:%S','now')
    where id=?
    '''

    CHECK_JOBS_STATUS_COMMAND = '''SELECT * FROM Job'''

    CHECK_JOB_DETAIL_COMMAND = '''SELECT * FROM JobDetail'''

    STATUS_MAP = {
        0: 'In Queue',
        1: 'Running',
        2: 'Completed',
        3: 'Failed',
    }

    def __init__(self, type_string):
        self.db_dir = os.path.join(self.BASE_DIR, type_string, 'db.sqlite3')

        if not os.path.exists(os.path.dirname(self.db_dir)):
            os.makedirs(os.path.dirname(self.db_dir))

        self.type_string = type_string

    def create_db(self):
        with sqlite3.connect(self.db_dir) as conn:
            conn.execute(self.CREATE_JOB_TABLE_COMMAND)
            conn.execute(self.CREATE_JOB_DETAIL_TABLE_COMMAND)
            conn.commit()

    def register_job(self, **kwargs):
        '''
        Registers jobs when it is valid. Returns primary key of job if valid, otherwise returns -1
        '''
        if self.LIMIT_RUN:
            job_ids = self._get_job_id_from_details(**kwargs)

            if len(job_ids) == 0:
                is_available = True
            else:
                is_available = self._check_job_availablity(job_ids)

        else:
            is_available = True

        if is_available:
            with sqlite3.connect(self.db_dir) as conn:
                cursor = conn.execute(self.ADD_JOB_COMMAND)
                conn.commit()
                primary_key = cursor.lastrowid
                self._register_job_details(primary_key, **kwargs)

            return primary_key
        else:
            return -1

    @abstractmethod
    def _register_job_details(self, primary_key, **kwargs):
        pass

    def update_job(self, id, status):
        with sqlite3.connect(self.db_dir) as conn:
            conn.execute(self.UPDATE_JOB_STATUS_COMMAND, (status, id))
            conn.commit()

    def read_jobs(self):
        with sqlite3.connect(self.db_dir) as conn:
            cursor = conn.execute(
                self.CHECK_JOBS_STATUS_COMMAND)

            details_cursor = conn.execute(self.CHECK_JOB_DETAIL_COMMAND)
            for row1, row2 in zip(cursor, details_cursor):
                print('Job ID: {}, Job Started: {}, Job Status: {}, Job Last Update: {}, Details: {}'.format(
                    row1[0], row1[1], self.STATUS_MAP[row1[2]], row1[3], row2))

    @abstractmethod
    def _get_job_id_statement(self, **kwargs):
        pass

    def _get_job_id_from_details(self, **kwargs):

        ids = []
        statement = self._get_job_id_statement(**kwargs)
        with sqlite3.connect(self.db_dir) as conn:
            cursor = conn.execute(statement)
            conn.commit()
            for row in cursor:
                ids.append(row[0])
        return ids

    def _check_job_availablity(self, job_ids):
        '''
        For jobs that should only be ran once
        '''

        job_id_string = '({})'.format(','.join(['?']*len(job_ids)))
        with sqlite3.connect(self.db_dir) as conn:

            cursor = conn.execute(
                self.CHECK_JOB_AVAILABILITY_COMMAND + job_id_string, job_ids)
            is_available = True
            for _ in cursor:
                is_available = False
                break
            return is_available
