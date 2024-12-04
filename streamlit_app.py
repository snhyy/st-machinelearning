import streamlit as st
import pandas as pd

st.title('Prediction App')

st.info('Predict Stock App')

with st.expander('Data'):
  st.write('Raw data')
  data = pd.read_csv('https://raw.githubusercontent.com/snhyy/data-test/refs/heads/main/online_retail_II.csv')
  data
