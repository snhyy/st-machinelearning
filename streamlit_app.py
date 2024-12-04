import streamlit as st
import pandas as pd

st.title('Prediction App')
st.info('Product recommendation')
st.sidebar.markdown('# Predict status')

uploaded_file = st.file_uploader("Upload file (CSV)", type=['csv'])
model_option = st.sidebar.selectbox("Select Machine Learning Model Training:", ["", "Decision Tree", "Logistic Regression", "Knn", "Support Vector Machine"])

if uploaded_file is not None:
    with st.expander('Data'):
        st.write('Raw data')
        data = pd.read_csv(uploaded_file)
        
    with st.expander('Top Product'):
        TopProducts= data.pivot_table(
        index=['StockCode','Description'],
        values='Quantity',
        aggfunc='sum').sort_values(
        by='Quantity', ascending=False)
        st.write(TopProducts.head(10))
    
        TopProducts.reset_index(inplace=True)
        st.bar_chart(TopProducts.head(10), y='Description', x='Quantity')
else:
    st.write('Waiting on file upload...') 