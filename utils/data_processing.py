import json
import os
import pickle
import time
from datetime import datetime

import pandas
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, insert, select


def request_raw_data():
    load_dotenv()
    key = os.getenv('API_KEY')
    request_url = f'https://slot-ml.com/api/v1/users/' \
                  f'{key}/vectors/?random'
    res = requests.get(request_url)
    if res.status_code != 200:
        return False

    return json.loads(res.text)


def gather_data(n):
    gathered_data = []
    for i in range(0, n):
        data = (request_raw_data())
        if not data:
            break
        save_vector(data)
        gathered_data.append(data)
        time.sleep(0.1)
    print(f'Downloaded and saved: {len(gathered_data)}')

    return gathered_data


def convert_save_dataframe(filename, lst):
    df = pandas.DataFrame(lst)
    cur_time = datetime.now().strftime("%H-%M-%S")
    with open(f'{filename}_{cur_time}_size {df.shape[0]}.pkl', 'wb') as file:
        pickle.dump(df, file)


def load_dataframe(filepath='.'):
    with open(filepath, 'rb') as f:
        df = pickle.load(f)
    if 'meta1' in df.columns and 'vector' in df.columns:
        return df
    else:
        raise ValueError('Incorrect dataframe')


def request_stats():
    load_dotenv()
    key = os.getenv('API_KEY')
    request_url = f'https://slot-ml.com//api/v1/users/{key}/stats'
    res = requests.get(request_url)
    if res.status_code != 200:
        print(res.text)
        return False
    return json.loads(res.text)


def upload_predictions(df):
    load_dotenv()
    api_key = os.getenv('API_KEY')
    url = f'https://slot-ml.com/api/v1/users/{api_key}/results/'

    for key in df['id']:
        value = df[df['id'] == key]['class'].values[0]
        myobj = {"vector": key, "class": value}
        x = requests.post(url, data=myobj)
        if x.status_code != 200:
            with st.spinner('Sleeping - sending error'):
                time.sleep(60)
        time.sleep(0.1)


def save_vector(vector):
    engine = create_engine("sqlite+pysqlite:///test_data/db.sqlite")
    metadata_obj = MetaData()
    user_table = Table("vectors", metadata_obj, autoload_with=engine)
    stmt = insert(user_table).values(**vector)
    with engine.connect() as conn:
        conn.execute(stmt)


def save_statistics():
    engine = create_engine("sqlite+pysqlite:///test_data/db.sqlite")
    metadata_obj = MetaData()
    user_table = Table("stats", metadata_obj, autoload_with=engine)

    for_insert = request_stats()['stats'][0]
    st.write(for_insert)
    for_insert['timestamp'] = datetime.now()

    last_stat = get_table_data('stats')

    if last_stat[-1][2] != for_insert['total_vectors']:
        stmt = insert(user_table).values(**for_insert)
        with engine.connect() as conn:
            conn.execute(stmt)
        st.write('Reload page')
    else:
        st.write('No updates, upload new vectors first')


def get_table_data(table_name):
    engine = create_engine("sqlite+pysqlite:///test_data/db.sqlite")
    metadata_obj = MetaData()
    user_table = Table(table_name, metadata_obj, autoload_with=engine)
    stmt = (select(user_table).order_by(user_table.c.id.asc()))
    with engine.connect() as conn:
        result = conn.execute(stmt)
        return result.all()


def save_model_name(model_name):
    engine = create_engine("sqlite+pysqlite:///test_data/db.sqlite")
    metadata_obj = MetaData()
    user_table = Table("models", metadata_obj, autoload_with=engine)
    stmt = insert(user_table).values(model_name=model_name)
    with engine.connect() as conn:
        conn.execute(stmt)
