import pickle
import time

import pandas as pd
import streamlit as st

from utils.data_processing import (convert_save_dataframe, gather_data,
                                   upload_predictions)
from utils.model import predict

st.sidebar.markdown("# Main page")

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
