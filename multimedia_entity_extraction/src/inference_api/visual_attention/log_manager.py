
from sqlalchemy import update, bindparam
import sqlite3


from inference_api.common.logdb.manager import LogDatabaseManager


class VALogDatabaseManager(LogDatabaseManager):

    def __init__(self, type_string):
        super().__init__(type_string)

    CREATE_JOB_DETAIL_TABLE_COMMAND = '''
        CREATE TABLE IF NOT EXISTS JobDetail(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_date TEXT NOT NULL)
        '''

    ADD_JOB_DETAIL_COMMAND = '''

        INSERT INTO JobDetail(id,batch_date) VALUES(?,?)

        '''

    LIMIT_RUN = True

    def _get_job_id_statement(self, **kwargs):
        statement = '''SELECT id FROM JobDetail WHERE batch_date='{}' '''.format(
            kwargs['batch_date'])
        return statement

    def _register_job_details(self, primary_key, **kwargs):
        statement = '''
        INSERT INTO JobDetail(id, batch_date) VALUES(? ,?)
        '''
        with sqlite3.connect(self.db_dir) as conn:
            cursor = conn.execute(self.ADD_JOB_DETAIL_COMMAND,
                                  (primary_key, kwargs['batch_date']))
            conn.commit()
