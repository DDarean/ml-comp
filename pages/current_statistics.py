import streamlit as st

from utils.data_processing import request_stats

st.markdown("# Current status")

stats = request_stats()
stats = stats['stats'][0]

for stat, value in stats.items():
    st.write(f'{stat}: {value}')
