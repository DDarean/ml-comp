import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_processing import get_statistics, save_statistics


def main():
    st.markdown("### Current status")
    stat = get_statistics()
    if get_statistics():
        df = pd.DataFrame(stat)
        options = ['avg_accuracy', 'avg_false_positive_ratio',
                   'avg_false_negative_ratio', 'avg_user_level',
                   'avg_spent_time', 'total_results', 'classify_data_ratio']
        param = st.selectbox('Select parameter', options=options)
        fig = px.line(df, x="total_vectors", y=param, markers=True)
        st.plotly_chart(fig)

        if st.button('Update stats'):
            save_statistics()
    else:
        st.write('Try again')


if __name__ == '__main__':
    main()
