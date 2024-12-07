import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

import random
from tqdm import tqdm
import plotly.express as px
from gensim.models import Word2Vec
import joblib
from joblib import load



st.title('abc App')
st.info('Sale analysis')

uploaded_file = st.file_uploader("Upload file (CSV)", type=['csv'])

if uploaded_file is not None:
    with st.expander('Data'):
        st.write('Review data')
        data = pd.read_csv(uploaded_file)
        data
        
    with st.expander('Top Product'):
        st.write('Top product by no of quantity:')
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
        
    with st.expander('Transaction'):
        st.write('Monthly transactions:')
        year = int(st.number_input('Year:', placeholder="Type the year...", format='%.0f'))
        
        # Convert "InvoiceDate" to datetime format if not already done
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        # Extract the year and month from the "InvoiceDate" column
        data['YearMonth'] = data['InvoiceDate'].dt.to_period('M')
        # Count the number of rows for each month
        monthly_counts = data.groupby('YearMonth').size()
        # Filter for the specified year
        monthly_counts_year = monthly_counts[monthly_counts.index.year == year]
        print(f"Total transactions: {monthly_counts_year.sum()}")

        # Plotting
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the bar plot
        a = sns.barplot(x=monthly_counts_year.index.astype(str), y=monthly_counts_year.values, palette='viridis', ax=ax)
        # Customize the plot
        ax.set_title(f'Monthly Transactions in {year}', fontsize=16)
        ax.set_xlabel('Month', fontsize=14)
        ax.set_ylabel('Number of Transactions', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)
    
    with st.expander('Product Recommendation:'):
        model = Word2Vec.load('../my_word2vec_model')
        
        products = data[["StockCode", "Description"]]
        products.drop_duplicates(inplace=True, subset='StockCode', keep="last")
        products_dict = products.groupby('StockCode')['Description'].apply(list).to_dict()
        def similar_products(v, n = 5):
            # extract most similar products
            ms = model.wv.most_similar([v], topn= n+1)[1:]
            # extract name and similarity score
            new_ms = []
            for j in ms:
                pair = (products_dict[j[0]][0], j[1])
                new_ms.append(pair)
            return new_ms  
        products_code = st.text_input('Product Recommended:', placeholder="Type the stockcode...")   
        if products_code is not None: 
            col1, col2 = st.columns(2) 
            with col1:
                try:
                    product_description = products_dict[products_code]
                    st.write('**Products Description**')
                    st.write(product_description)
                except KeyError:
                    st.error(f"Product code '{products_code}' not found in dictionary.")
            with col2:
                try:
                    similar_items = similar_products(model.wv[products_code])
                    st.write('**Similar Products**')
                    st.write(similar_items)
                except KeyError:
                    st.error(f"Product code '{products_code}' not found in similar products.")
        else:
            st.write('Waiting for stockcode...') 
        
        
else:
    st.write('Waiting for file upload...') 