import pandas as pd
import streamlit as st

from utils.data_processing import get_table_data, save_model_name
from utils.model import Kmeans, Preprocessor


def main():
    st.markdown("### Train model")

    if st.button(label='Train on the latest data'):
        model = Kmeans()
        preprocessor = Preprocessor()
        data = get_table_data('vectors')
        data = pd.DataFrame(data)
        preprocessor.fit_save_vectorizers(data)
        data = preprocessor.transform_data(data)
        model.fit_save_model(data)
        save_model_name(model.model_name)
        st.write('Model retrained and saved in "models" folder')


if __name__ == '__main__':
    main()
