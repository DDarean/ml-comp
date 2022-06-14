import time

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_processing import (convert_save_dataframe, gather_data,
                                   get_table_data, upload_predictions)
from utils.model import Preprocessor
import pickle, pandas

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
name = st.selectbox(label='Select model',
                          options=models_list['model_name'].unique())


if st.button(label='Load predictions'):
    model_name = 'models/KMeans-2950.pkl'
    vectorizer_meta = f'models/vectorizers/vector-meta-2950.pkl'
    vectorizer_vector = f'models/vectorizers/vector-vector2950.pkl'

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
            convert_save_dataframe(f'{i}-cycle5', gather_5)
            preprocessor = Preprocessor()
            df = pandas.DataFrame(gather_5)
            data = preprocessor.transform_data(df, vectorizer_meta, vectorizer_vector, load=True)
            with open(model_name, 'rb') as f:
                model = pickle.load(f)
            pred = model.pred(df, data)
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
