import json
import pickle
import time
from datetime import datetime

import pandas
import requests


def request_raw_data():
    request_url = 'https://slot-ml.com/api/v1/users/' \
                  'eade5a348b246aa623e21fd044863764247b1438/vectors/?random'
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
        time.sleep(0.3)
    print(f'Downloaded: {len(gathered_data)}')

    return gathered_data


def convert_save_dataframe(filename, lst):
    df = pandas.DataFrame(lst)
    cur_time = datetime.now().strftime("%H-%M-%S")
    with open(f'{filename}_{cur_time}_size {df.shape[0]}.pkl', 'wb') as file:
        pickle.dump(df, file)
