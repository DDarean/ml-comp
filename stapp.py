import os
import pickle
import time

import pandas
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from dotenv import load_dotenv

from utils.data_processing import (gather_data,
                                   get_table_data, upload_predictions)
from utils.models import Autoencoder, Preprocessor

load_dotenv()
model_path = os.getenv('MODEL_PATH')
vectorizer_path = os.getenv('VECTORIZER_PATH')
model_name = os.getenv('DEFAULT_MODEL_NAME')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(vectorizer_path):
    os.makedirs(vectorizer_path)

st.markdown('### Accuracy history')
st.markdown('For detailed stats refer page "Current statistics"')

stat = get_table_data('stats')
df = pd.DataFrame(stat)
fig = px.line(df, x='total_vectors', y='avg_accuracy',
              markers=True, width=600, height=400)
st.plotly_chart(fig)

st.markdown('### Gather new vectors')

num_iter = st.text_input(label='number of iterations')
num_vectors = st.text_input(label='number of vectors per iteration')

models_list = pd.DataFrame(get_table_data('models'))
if models_list.shape[0] == 0:
    st.warning('Please train a model first')
    st.stop()
name = st.selectbox(label='Select model',
                    options=models_list['model_name'].unique())
num_of_vectors = int(
    models_list[models_list['model_name'] == name]['num_of_vectors'].unique())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if st.button(label='Load predictions'):
    kmeans_path = f'{model_path}/{model_name}-KMeans-{num_of_vectors}.pkl'
    ae_path = f'{model_path}/{model_name}-AE-{num_of_vectors}.pkl'
    vectorizer_meta = f'{vectorizer_path}/{model_name}-meta-{num_of_vectors}.pkl'
    vectorizer_vector = f'{vectorizer_path}/{model_name}-vector-{num_of_vectors}.pkl'

    if num_iter and num_vectors:
        num_iter = int(num_iter)
        num_vectors = int(num_vectors)
        message = st.empty()
        total_processed = 0
        for i in range(num_iter):
            gather_5 = gather_data(num_vectors)
            total_processed += len(gather_5)
            if len(gather_5) == 0:
                with st.spinner('Sleeping'):
                    time.sleep(1800)
                continue
            # convert_save_dataframe(f'{i}-cycle5', gather_5)
            preprocessor = Preprocessor()
            df = pandas.DataFrame(gather_5)
            data = preprocessor.transform_data(df, vectorizer_meta,
                                               vectorizer_vector, load=True)
            with open(kmeans_path, 'rb') as f:
                model = pickle.load(f)
            model_ae = Autoencoder(input_shape=data.shape[1])
            model_ae.load_state_dict(torch.load(ae_path))
            model_ae.to(device)
            model_ae.eval()
            data = data.toarray()
            encoded = model_ae.encode(
                torch.from_numpy(data).float().to(device))
            detached = encoded.cpu().detach().numpy()
            pred = model.pred(df, detached)
            upload_predictions(pred)
            with message.container():
                st.write(f'Iteration {i} complete')
                st.write(f'Vectors uploaded: {total_processed}')
        st.write('DONE')
        st.write(f'Number of uploaded vectors: {total_processed}')
    else:
        st.write('Enter num of iterations')

else:
    st.write('Start and upload predictions')
