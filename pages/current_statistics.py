import pandas as pd
import plotly.express as px
import streamlit as st

from utils.utils import get_statistics, save_statistics


def main():
    st.markdown("# Current status")
    stat = get_statistics()
    if get_statistics():
        df = pd.DataFrame(stat)
        param = st.selectbox('Select parameter', options=[col for col
                                                          in df.columns
                                                          if col not in
                                                          ['id', 'timestamp']])
        fig = px.line(df, x="timestamp", y=param, markers=True)
        st.plotly_chart(fig)

        if st.button('Update stats'):
            save_statistics()
    else:
        st.write('Try again')

if __name__ == '__main__':
    main()
