import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.data_processing import get_table_data, save_model_name
from utils.models import Kmeans, Preprocessor, TrainerAE

load_dotenv()
model_name = os.getenv('DEFAULT_MODEL_NAME')


def main():
    st.markdown("### Train model")

    if st.button(label='Train on the latest data'):
        model = Kmeans()
        preprocessor = Preprocessor()
        data = get_table_data('vectors')
        data = pd.DataFrame(data)
        preprocessor.fit_save_vectorizers(data)
        data_transformed = preprocessor.transform_data(data)
        model_ae = TrainerAE(data_transformed)
        with st.spinner('Training autoencoder'):
            model_ae.fit_save_model(num_of_vectors=data.shape[0])
        st.write('AE trained')
        with st.spinner('Training KMeans classifier'):
            model.fit_save_model(model_ae.encode(data_transformed))
        save_model_name(model_name, num_of_vectors=data.shape[0])
        st.write('Models trained and saved in "models" folder')


if __name__ == '__main__':
    main()
