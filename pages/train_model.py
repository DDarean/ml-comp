import pandas as pd
import streamlit as st

from utils.data_processing import get_latest_data, save_model_name
from utils.model import Kmeans


def main():
    st.markdown("### Train model")

    if st.button(label='Train on the latest data'):
        model = Kmeans()
        data = get_latest_data()
        df = pd.DataFrame(data)
        model.fit_save_model(df)
        save_model_name(model.model_name)
        st.write('Model retrained and saved in "models" folder')


if __name__ == '__main__':
    main()
