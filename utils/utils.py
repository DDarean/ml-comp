from datetime import datetime

from data_processing import request_stats
from sqlalchemy import MetaData, Table, create_engine, insert, select


def save_statistics():
    engine = create_engine("sqlite+pysqlite:///../test_data/db.sqlite")
    metadata_obj = MetaData()
    user_table = Table("stats", metadata_obj, autoload_with=engine)

    for_insert = request_stats()['stats'][0]
    for_insert['timestamp'] = datetime.now()

    last_stat = get_statistics(get_all=True)

    if last_stat['total_vectors'] != for_insert['total_vectors']:
        stmt = insert(user_table).values(**for_insert)
        with engine.connect() as conn:
            conn.execute(stmt)
    else:
        print('no updates')


def get_statistics(get_all=True):
    engine = create_engine("sqlite+pysqlite:///../test_data/db.sqlite")
    metadata_obj = MetaData()
    user_table = Table("stats", metadata_obj, autoload_with=engine)
    stmt = (select(user_table).order_by(user_table.c.id.asc()))
    with engine.connect() as conn:
        result = conn.execute(stmt)
        if type == get_all:
            return result.fetchall()
        else:
            return result.fetchall()[-1]
