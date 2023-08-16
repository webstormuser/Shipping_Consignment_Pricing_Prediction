import pandas as pd
from shipping_prediction.logger import logging
from shipping_prediction.exception import ShippingException
from shipping_prediction.config import mongo_client
import os,sys
import yaml
import numpy as np
import pickle 
import re 
def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        logging.info("Replacing column space by underscore")
        new_columns = [col.replace(' ', '_') for col in df.columns]
        df = df.rename(columns=dict(zip(df.columns, new_columns)))
        logging.info(f"Columns in dataframe : {df.columns}")
        logging.info(f"{'>>'*20}*{'<<'*20}")
        for col in df.columns:
                logging.info(f" {col} : {df[col].dtype}")

        logging.info(f"Reordering column ASN/DN and PO_/_SO_# since it contain two values")
        df['ASN/DN_#']=df['ASN/DN_#'].apply(reorder)
        df['PO_/_SO_#']=df['PO_/_SO_#'].apply(reorder)

        logging.info(f"Converting PQ_First_Sent_to_Client_Date column into Date format")
        df['PQ_First_Sent_to_Client_Date']=pd.to_datetime(df['PQ_First_Sent_to_Client_Date'],errors='coerce',format='%m/%d/%y')
        df['PQ_First_Sent_to_Client_Date'].fillna(df['PQ_First_Sent_to_Client_Date'].min(),inplace=True)
        df['PQ_First_Sent_to_Client_Date']=df['PQ_First_Sent_to_Client_Date'].dt.strftime('%d-%b-%y')
        df['PQ_First_Sent_to_Client_Date']=pd.to_datetime(df['PQ_First_Sent_to_Client_Date'], format='%d-%b-%y')

        logging.info(f"Applying date transformation on date columns to change datatype from object to date")
        df['Scheduled_Delivery_Date']=df['Scheduled_Delivery_Date'].apply(transform_date)
        df['Delivered_to_Client_Date']=df['Delivered_to_Client_Date'].apply(transform_date)
        df['Delivery_Recorded_Date']=df['Delivery_Recorded_Date'].apply(transform_date)

        logging.info(f"Applying weight value filling to column")
        df['Weight_(Kilograms)']= df['Weight_(Kilograms)'].apply(extract_weight)
        df['Weight_(Kilograms)']=df['Weight_(Kilograms)'].apply(lambda x: np.nan if x == 'Weight Captured Separately' else x)
        df['Freight_Cost_(USD)']=df['Freight_Cost_(USD)'].apply(trans_freight_cost)
        df['Freight_Cost_(USD)']=df['Freight_Cost_(USD)'].astype('float')
        
        
        logging.info(f"from columns Delivery_Recorded_Date and PQ_First_Sent_to_Client_Date making new column days_to_process")
        df['Days_to_Process']=df['Delivery_Recorded_Date']-df['PQ_First_Sent_to_Client_Date']
        df['Days_to_Process']=df['Days_to_Process'].dt.days.astype('int64')
                
        for col in df.columns:
                logging.info(f" {col} : {df[col].dtype}")
        return df
    except Exception as e:
        raise ShippingException(e, sys)


def reorder(data):
    '''This function reorders the columns based on their unique values 
    '''
    data_split=data.split("-")
    data_return=data_split[0]
    return data_return
    
def transform_date(data):
    '''
    This function applies transformation of object to date of columns Scheduled_Delivery_Date ,Delivered_to_Client_Date,PO_First_Sent_to_Vendor_Date
    '''
    data=data.replace("-","-")
    data=pd.to_datetime(data, format="%d-%b-%y")
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


def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise ShippingException(e, sys)

