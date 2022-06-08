import pickle
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_processing import (convert_save_dataframe, gather_data,
                                   upload_predictions)
from utils.model import predict
from utils.utils import get_statistics

st.markdown('### Accuracy history')
st.markdown('For detailed stats refer page "Current statistics"')

stat = get_statistics()
df = pd.DataFrame(stat)
fig = px.line(df, x='timestamp', y='avg_accuracy', markers=True, width=600, height=400)
st.plotly_chart(fig)

st.markdown('### Gather new vectors')

num_iter = st.text_input(label='number of iterations')
num_vectors = st.text_input(label='number of vectors per iteration')


if st.button(label='Load predictions'):
    with open('../experiments/model-new.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('../experiments/vectorizer_meta-new.pkl', 'rb') as f:
        vectorizer_meta = pickle.load(f)

    with open('../experiments/vectorizer_vector-new.pkl', 'rb') as f:
        vectorizer_vector = pickle.load(f)

    st.write('Model and vectorizers loaded')

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
            pred = predict(pd.DataFrame(gather_5), vectorizer_meta,
                           vectorizer_vector, model)
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
