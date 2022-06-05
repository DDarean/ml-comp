import json
import os
import pickle
import time
from datetime import datetime

import pandas
import requests
from dotenv import load_dotenv


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
        gathered_data.append(data)
        time.sleep(0.1)
    print(f'Downloaded: {len(gathered_data)}')

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
