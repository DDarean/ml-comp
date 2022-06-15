import pandas as pd
import streamlit as st

from utils.data_processing import get_table_data, save_model_name
from utils.models import Kmeans, Preprocessor, TrainerAE


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
        model_ae.fit_save_model()
        st.write('AE trained')
        model.fit_save_model(model_ae.encode(data_transformed))
        save_model_name('AE+KMeans', num_of_vectors=data.shape[0])
        st.write('Model retrained and saved in "models" folder')


if __name__ == '__main__':
    main()
