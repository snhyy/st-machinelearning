import streamlit as st
import pandas as pd
import random

st.title('Prediction App')
st.info('Product recommendation')
st.sidebar.markdown('# Predict status')

uploaded_file = st.file_uploader("Upload file (CSV)", type=['csv'])

if uploaded_file is not None:
    st.write("Upload Success:")
    with st.expander('Data'):
        st.write('Review data')
        data = pd.read_csv(uploaded_file)
        data
        
    with st.expander('Top Product'):
        st.write('Top product by no of quantity')
        TopProducts= data.pivot_table(
        index=['StockCode','Description'],
        values='Quantity',
        aggfunc='sum').sort_values(
        by='Quantity', ascending=False)
        TopProducts.reset_index(inplace=True)
        st.bar_chart(TopProducts.head(10), y='Description', x='Quantity')
        
        st.write('Top product by no of customers')
        customers = data["CustomerID"].unique().tolist()
        CustomersBoughts = data.pivot_table(index=['StockCode','Description'],
                                values='CustomerID',
                                aggfunc=lambda x: len(x.unique())).sort_values(by='CustomerID', ascending=False)
        CustomersBoughts.reset_index(inplace=True)
        st.bar_chart(CustomersBoughts.head(10), y='Description', x='CustomerID')
        
else:
    st.write('Waiting on file upload...') 
