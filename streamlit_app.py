import streamlit as st
import pandas as pd
import random
from tqdm import tqdm
from gensim.models import Word2Vec

st.title('Prediction App')
st.info('Product recommendation')
st.sidebar.markdown('# Predict status')

uploaded_file = st.file_uploader("Upload file (CSV)", type=['csv'])

if uploaded_file is not None:
    with st.expander('Data'):
        st.write('Review data')
        data = pd.read_csv(uploaded_file)
        
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
        
    with st.expander('Recommend product'):
        # split data
        random.shuffle(customers)
        # extract 90% of customer ID's
        customers_train = [customers[i] for i in range(round(0.9*len(customers)))]
        # split data into train and validation set
        train_df = data[data['CustomerID'].isin(customers_train)]
        validation_df = data[~data['CustomerID'].isin(customers_train)]
        
        purchases_train = []
        for i in tqdm(customers_train):
            temp = train_df[train_df["CustomerID"] == i]["StockCode"].tolist()
            purchases_train.append(temp)
        
        purchases_val = []
        for i in tqdm(validation_df['CustomerID'].unique()):
            temp = validation_df[validation_df["CustomerID"] == i]["StockCode"].tolist()
            purchases_val.append(temp)
            
        model = Word2Vec(window = 15, sg = 1, hs = 0,
                 negative = 10, 
                 alpha=0.03, min_alpha=0.0007,
                 seed = 121)

        model.build_vocab(purchases_train, progress_per=200)
        model.train(purchases_train, total_examples = model.corpus_count, epochs=10, report_delay=1)
        
        st.write(model)
else:
    st.write('Waiting on file upload...') 