import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

st.title('Prediction App')
st.info('Product recommendation')
st.sidebar.markdown('# Predict status')

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
        year = st.number_input('Year:', value=None, placeholder="Type a number...", format='%0f')
        
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
        
    with st.expander('Customer Segmentation'):
        st.write('Cứu tui')
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
        data = data[~data["Invoice"].str.contains("C", na=False)]
        today_date = dt.datetime(2011, 12, 11)
        
        rfm = data.groupby('CustomerID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'Total': lambda Total: Total.sum()})
        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm = rfm[rfm["monetary"] > 0]
        
        rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
        st.write('The most frequent and most recent customers')
        rfm[rfm["RFM_SCORE"] == "55"]
        st.write('The least frequent and least recent customers')
        rfm[rfm["RFM_SCORE"] == "11"]
        st.write('Customer Segmentation Using RFM Analysis:')
        seg_map = {
            r'[1-2][1-2]': 'hibernating',
            r'[1-2][3-4]': 'at_risk',
            r'[1-2]5': 'cant_loose',
            r'3[1-2]': 'about_to_sleep',
            r'33': 'need_attention',
            r'[3-4][4-5]': 'loyal_customers',
            r'41': 'promising',
            r'51': 'new_customers',
            r'[4-5][2-3]': 'potential_loyalists',
            r'5[4-5]': 'champions'
            }
        rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True) 
        rfm[['segment', 'recency', 'frequency', 'monetary']].groupby('segment').agg(['mean', 'count'])
        
else:
    st.write('Waiting on file upload...') 