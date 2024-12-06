import streamlit as st
import pandas as pd
import numpy as np

import lifetimes
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import *
from lifetimes.plotting import *
from sklearn.metrics import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import datetime as dt


uploaded_file = st.file_uploader("Upload file (CSV)", type=['csv'])

st.sidebar.markdown('## Predict credit status')
st.sidebar.header('Select model')
model_option = st.sidebar.selectbox('Select model:', ['BG/NBD','Gamma-Gamma'])

if uploaded_file is not None:
    with st.expander('Data'):
        st.write('Review data')
        data = pd.read_csv(uploaded_file)
        data
    
    with st.expander('RFM Analysis:'):
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
        
        st.write('Customer segmentation')
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
        st.write(rfm[['segment', 'recency', 'frequency', 'monetary']].groupby('segment').agg(['mean', 'count']))
    st.write('BG/NBD Model for Predicting the Number of Purchase')
    st.write('Gamma-Gamma Model for Estimating CLV')
    
    
    with st.expander('Prediction'):  
        # Define of end of observation period
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate']).dt.date
        last_global_purchase = data['InvoiceDate'].max()

        # Define end of calibration period
        n_months = 6
        cal_period_end = last_global_purchase - dt.timedelta(days=(n_months * 30))

        # Calibration period RFM data
        clv_cal = summary_data_from_transaction_data(
            data, 
            'CustomerID', 'InvoiceDate', 'Total',
            observation_period_end=cal_period_end,
            include_first_transaction=False
            )

        # Holdout period RFM data
        clv_holdout = summary_data_from_transaction_data(
            data[data['InvoiceDate'] > cal_period_end], 
            'CustomerID', 'InvoiceDate', 'Total',
            observation_period_end=last_global_purchase,
            include_first_transaction=True
            )

        # Combine two the DataFrame
        cal_and_holdout_data = clv_cal.copy()
        cal_and_holdout_data = cal_and_holdout_data.rename(
            columns={'recency': 'recency_cal', 
                'frequency': 'frequency_cal', 
                'monetary_value': 'monetary_value_cal', 
                'T': 'T_cal'})
        cal_and_holdout_data['frequency_holdout'] = clv_holdout['frequency']
        cal_and_holdout_data['monetary_value_holdout'] = clv_holdout['monetary_value']
        cal_and_holdout_data['duration_holdout'] = n_months * 30
        # We only need customers who have repeat purchase
        cal_and_holdout_data = cal_and_holdout_data[cal_and_holdout_data['frequency_cal'] > 0].fillna(0)
        
        best_rmse = 1e+9
        for coef in range(1, 11):
            coef = coef / 100
            bgf = BetaGeoFitter(penalizer_coef=coef)
            # Fit the model with calibration period data
            bgf.fit(cal_and_holdout_data['frequency_cal'], 
            cal_and_holdout_data['recency_cal'], 
            cal_and_holdout_data['T_cal'])
            pred_freq = pd.DataFrame(
            bgf.conditional_expected_number_of_purchases_up_to_time(
                    cal_and_holdout_data['duration_holdout'], # Number of days to predict
                    cal_and_holdout_data['frequency_cal'], 
                    cal_and_holdout_data['recency_cal'], 
                    cal_and_holdout_data['T_cal']
                ), columns=['pred_freq']).reset_index()
            new_df = cal_and_holdout_data.reset_index().merge(pred_freq, on='CustomerID').dropna()
            rmse = np.sqrt(mean_squared_error(new_df['frequency_holdout'], new_df['pred_freq']))
            if rmse < best_rmse:
                 best_rmse = rmse
                 best_coef = coef
        if model_option == 'BG/NBD':
            st.markdown('## BG/NBD')
            bgf = BetaGeoFitter(penalizer_coef=best_coef)
            bgf.fit(
                cal_and_holdout_data['frequency_cal'], 
                cal_and_holdout_data['recency_cal'], 
                cal_and_holdout_data['T_cal']
                )
            cal_and_holdout_data[f'exp_purchases_next_{n_months}m'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                n_months * 30, # Number of days to predict
                cal_and_holdout_data['frequency_cal'],
                cal_and_holdout_data['recency_cal'], 
                cal_and_holdout_data['T_cal']
                )
            actual_purchase = cal_and_holdout_data['frequency_holdout'].sum()
            pred_purchase = cal_and_holdout_data[f'exp_purchases_next_{n_months}m'].sum()
            percentage_error_purchase = abs(pred_purchase - actual_purchase) / actual_purchase
            st.write(f"""
            Number of of purchases for the next {n_months} months...
            - Actual: {actual_purchase.astype(int)}
            - Predited: {pred_purchase.astype(int)}""")
            st.write(f"Percentage error: {percentage_error_purchase:.3%}\n")
            fig, ax = plt.subplots()
            lifetimes.plotting.plot_calibration_purchases_vs_holdout_purchases(bgf, cal_and_holdout_data, n=15, ax=ax)

            # Display the plot in Streamlit
            st.pyplot(fig)
        if model_option == 'Gamma-Gamma':
            st.markdown('## Gamma-Gamma')
            cal_and_holdout_data['clv_holdout'] = (
                cal_and_holdout_data['frequency_holdout'] * cal_and_holdout_data['monetary_value_holdout']
            )
            best_rmse = 1e+9
            for discount_rate in range(1, 11):
                discount_rate = discount_rate / 100
                # Fit the model with calibration period data
                ggf = GammaGammaFitter(penalizer_coef=coef)
                ggf.fit(
                    cal_and_holdout_data['frequency_cal'],
                    cal_and_holdout_data['monetary_value_cal']
                )
                pred_clv = pd.DataFrame(
                    ggf.customer_lifetime_value(
                        bgf,
                        cal_and_holdout_data['frequency_cal'],
                        cal_and_holdout_data['recency_cal'],
                        cal_and_holdout_data['T_cal'],
                        cal_and_holdout_data['monetary_value_cal'],
                        time=n_months, # Number of months to predict
                        freq='D',
                        discount_rate=discount_rate
                    )
                ).reset_index()
                new_df = cal_and_holdout_data.reset_index().merge(pred_clv, on='CustomerID').dropna()
                new_df = new_df.rename(columns={'clv': 'pred_clv'})
                rmse = np.sqrt(mean_squared_error(new_df['clv_holdout'], new_df['pred_clv']))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_coef = coef
                    best_discount_rate = discount_rate   
                       
            ggf = GammaGammaFitter(penalizer_coef=best_coef)
            ggf.fit(
                cal_and_holdout_data['frequency_cal'],
                cal_and_holdout_data['monetary_value_cal']
            ) 
            cal_and_holdout_data[f'exp_clv_next_{n_months}m'] = ggf.customer_lifetime_value(
                bgf,
                cal_and_holdout_data['frequency_cal'],
                cal_and_holdout_data['recency_cal'],
                cal_and_holdout_data['T_cal'],
                cal_and_holdout_data['monetary_value_cal'],
                time=n_months, # Number of months to predict
                freq='D',
                discount_rate = best_discount_rate
            )
            rmse = np.sqrt(mean_squared_error(
                cal_and_holdout_data['clv_holdout'], 
                cal_and_holdout_data[f'exp_clv_next_{n_months}m']
            ))    
            actual_clv = cal_and_holdout_data['clv_holdout'].sum()
            pred_clv = cal_and_holdout_data[f'exp_clv_next_{n_months}m'].sum()
            percentage_error_clv = abs(pred_clv - actual_clv) / actual_clv
            st.write(f"""
            Revenue for the next {n_months} months:
            - Actual: {actual_clv.astype(int)}
            - Predicted: {pred_clv.astype(int)}""")
            st.write(f"Percentage error: {percentage_error_clv:.3%}")

            def plot_calibration_clv_vs_holdout_clv(calibration_holdout_matrix, n_months):
                data = calibration_holdout_matrix.copy()
                data['clv_cal'] = ((data['frequency_cal'] * data['monetary_value_cal']) / 1000).round(1)
                data['clv_holdout'] = (data['clv_holdout'] / 1000).round(1)
                data[f'exp_clv_next_{n_months}m'] = (data[f'exp_clv_next_{n_months}m'] / 1000).round(1)

                fig, ax = plt.subplots(figsize=(8, 5))
                data.groupby('clv_cal')[['clv_holdout', f'exp_clv_next_{n_months}m']].mean().plot(
                ax=ax,
                xlabel='CLV (thousand of dollar) in Calibration Period',
                ylabel='Average of CLV (thousand of dollar) in Holdout Period',
                title='Actual CLV in Holdout Period vs Predicted CLV'
                )
                return fig
                # Assuming 'cal_and_holdout_data' and 'n_months' are defined
            fig = plot_calibration_clv_vs_holdout_clv(cal_and_holdout_data, n_months)
            st.pyplot(fig)
            
            clv_segment = cal_and_holdout_data.copy()
            clv_segment['segment'] = pd.qcut(
                clv_segment[f'exp_clv_next_{n_months}m'], 
                4,
                labels=['Churned', 'Need Attention', 'Loyal Customers', 'Champions']
            )
            
            def plot_holdout_purchase_vs_prob_alive(calibration_holdout_matrix, bgf):
                data = calibration_holdout_matrix.copy().reset_index()
                data['prob_alive'] = bgf.conditional_probability_alive(
                    data['frequency_cal'], 
                    data['recency_cal'], 
                    data['T_cal']).round(1)
                data['have_purchased'] = data['frequency_holdout'].apply(lambda x: 'yes' if x > 0 else 'no')
                plot_data = data.groupby(['prob_alive', 'have_purchased']).size() / data.groupby('prob_alive').size()
                plot_data = plot_data.reset_index(name='ratio')
                fig, ax = plt.subplots()
                sns.barplot(data=plot_data, x='prob_alive', y='ratio', hue='have_purchased', ax=ax)
                ax.set_title('Probability of Alive vs Ratio of Customers Who Have Actual Purchase')
                ax.set_xlabel('Probability of Alive')
                ax.set_ylabel('Ratio of Customers')
                return fig

            # Assuming 'cal_and_holdout_data' and 'bgf' are defined
            fig = plot_holdout_purchase_vs_prob_alive(cal_and_holdout_data, bgf)
            st.pyplot(fig) 
            st.write('**Note:** can see in the above bar plot, as the probability of alive decreases, the ratio of customers who have actual purchase descreases. For those probability threshold where the ratio of customers who have and have no actual purchase are approximated, businesses can consider improve customer retention by retargeting these customers.')     
else:
    st.write('Waiting on file upload...') 