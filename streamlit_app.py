import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Prediction App')
st.info('Product recommendation')
st.sidebar.markdown('# Predict status')

uploaded_file = st.file_uploader("Upload file (CSV)", type=['csv'])
model_option = st.sidebar.selectbox("Select Machine Learning Model Training:", ["", "Decision Tree", "Logistic Regression", "Knn", "Support Vector Machine"])

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
        ax.set_title(f'Monthly Transactions in {year:.0f}', fontsize=16)
        ax.set_xlabel('Month', fontsize=14)
        ax.set_ylabel('Number of Transactions', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

        
        
        
    with st.expander('Recommend product'):
        # split data
        st.write('cuwus tui')
        
        
        
else:
    st.write('Waiting on file upload...') 