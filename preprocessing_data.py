import pymongo
import pandas as pd
import json
import re
import numpy as np


df=pd.read_csv("SCMS_Delivery_History_Dataset.csv")
df.columns = [col.replace(' ', '_') for col in df.columns]
print(df.shape)
print(df.columns)
print(df.dtypes)

print("preprocessing dataframe and saving into clean CSV")

# Utility function to preprocess DataFrame columns
def preprocess_dataframe(df):
    """
    This function preprocesses DataFrame columns
    """
    df['ASN/DN_#'] = df['ASN/DN_#'].apply(reorder)
    df['PO_/_SO_#'] = df['PO_/_SO_#'].apply(reorder)

    df['PQ_First_Sent_to_Client_Date'] = pd.to_datetime(df['PQ_First_Sent_to_Client_Date'], errors='coerce', format='%m/%d/%y')
    df['PQ_First_Sent_to_Client_Date'].fillna(df['PQ_First_Sent_to_Client_Date'].min(), inplace=True)
    df['PQ_First_Sent_to_Client_Date'] = df['PQ_First_Sent_to_Client_Date'].dt.strftime('%d-%b-%y')
    df['PQ_First_Sent_to_Client_Date'] = pd.to_datetime(df['PQ_First_Sent_to_Client_Date'], format='%d-%b-%y')
    df['Delivery_Recorded_Date'] = df['Delivery_Recorded_Date'].apply(transform_date)
    df['Days_to_Process'] = df['Delivery_Recorded_Date'] - df['PQ_First_Sent_to_Client_Date']
    df['Days_to_Process'] = df['Days_to_Process'].dt.days.astype('int64')

    df['Scheduled_Delivery_Date'] = df['Scheduled_Delivery_Date'].apply(transform_date)
    df['Delivered_to_Client_Date'] = df['Delivered_to_Client_Date'].apply(transform_date)
    

    df['Weight_(Kilograms)'] = df['Weight_(Kilograms)'].apply(extract_weight)
    df['Weight_(Kilograms)'] = df['Weight_(Kilograms)'].apply(lambda x: np.nan if x == 'Weight Captured Separately' else x)
    df['Freight_Cost_(USD)'] = df['Freight_Cost_(USD)'].apply(trans_freight_cost)
    df['Freight_Cost_(USD)'] = df['Freight_Cost_(USD)'].astype('float')

    return df


def reorder(data):
    '''This function reorders the columns based on their unique values 
    '''
    data_split=data.split("-")
    data_return=data_split[0]
    return data_return
    
def transform_date(data):
    '''
    This function applies transformation of object to date of columns Scheduled_Delivery_Date, Delivered_to_Client_Date, PO_First_Sent_to_Vendor_Date
    '''
    data = pd.to_datetime(data, format="%d-%b-%y")
    return data
    
def trans_freight_cost(x):
    if x.find("See")!=-1:
        return np.nan
    elif x=="Freight Included in Commodity Cost" or x=="Invoiced Separately":
        return 0
    else:
        return x

def extract_weight(value):
    '''This function exctracts weight value from given in text description format
    '''
    match = re.search(r'\d+\.*\d*', value)  # Extract numbers (including decimals)
    if match:
        return float(match.group())
    else:
        return value

df=preprocess_dataframe(df)
print(df.shape)
print(df.columns)
print(df.dtypes)
df.to_csv("clean_SCMS_Delivery_History_Dataset.csv",index=False)