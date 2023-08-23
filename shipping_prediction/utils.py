import pandas as pd
from shipping_prediction.logger import logging
from shipping_prediction.exception import ShippingException
from shipping_prediction.config import mongo_client
import os
import sys
import yaml
import numpy as np
import pickle
import joblib
import re
from sklearn.model_selection import train_test_split


# Function to fetch a collection from MongoDB as a DataFrame
def get_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Description: This function returns a collection as a DataFrame
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    Returns:
    Pandas DataFrame of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        # Drop the "_id" column if present
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id")
            df = df.drop("_id", axis=1)
        logging.info(f"Columns in dataframe : {df.columns}")
        logging.info(f"{'>>'*20}*{'<<'*20}")
        for col in df.columns:
                logging.info(f" {col} : {df[col].dtype}")
        return df
    except Exception as e:
        raise ShippingException(e, sys)

'''
def typo_check(valid_function_names):
    try:
        def decorator(func):
            def wrapper(*args, **kwargs):
                function_name = func.__name__
                if function_name not in valid_function_names:
                    print(f"Warning: Possible typo in function name '{function_name}'")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    except Exception as e:
        raise ShippingException(e,sys)
'''


def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise ShippingException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array in file in .npz format
    file_path is the location of file where to store numpy array
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        np.save(file_path, array=array)
    except Exception as e:
        raise ShippingException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise ShippingException(e, sys) from e


def load_num_array(file_path: str) -> np.array:
    """
    Load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        return np.load(file_path)
    except Exception as e:
        raise ShippingException(e, sys) from e


def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise ShippingException(e, sys) from e












      
