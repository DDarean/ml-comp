from datetime import datetime

import streamlit as st
from sqlalchemy import MetaData, Table, create_engine, insert, select

from utils.data_processing import request_stats


def save_statistics():
    engine = create_engine("sqlite+pysqlite:///test_data/db.sqlite")
    metadata_obj = MetaData()
    user_table = Table("stats", metadata_obj, autoload_with=engine)

    for_insert = request_stats()['stats'][0]
    st.write(for_insert)
    for_insert['timestamp'] = datetime.now()

    last_stat = get_statistics()

    if last_stat[-1][2] != for_insert['total_vectors']:
        stmt = insert(user_table).values(**for_insert)
        with engine.connect() as conn:
            conn.execute(stmt)
    else:
        st.write('No updates')


def get_statistics():
    engine = create_engine("sqlite+pysqlite:///test_data/db.sqlite")
    metadata_obj = MetaData()
    user_table = Table("stats", metadata_obj, autoload_with=engine)
    stmt = (select(user_table).order_by(user_table.c.id.asc()))
    with engine.connect() as conn:
        result = conn.execute(stmt)
        return result.all()

