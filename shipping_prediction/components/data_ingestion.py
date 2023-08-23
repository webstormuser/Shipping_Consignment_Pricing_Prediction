from shipping_prediction import utils
from  shipping_prediction.exception import ShippingException
from  shipping_prediction.logger import logging
from  shipping_prediction.entity import artifact_entity 
from shipping_prediction.entity import config_entity
import os,sys
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ShippingException(error_message=e, error_detail=sys)
            logging.ERROR(e)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame  = utils.get_collection_as_dataframe(
            database_name=self.data_ingestion_config.database_name, 
            collection_name=self.data_ingestion_config.collection_name) 

            logging.info(f" Shape of df {df.shape}")

            logging.info("Save data in feature store")
           
            #checking any duplicate record inside the dataset or not
            logging.info(f"Any duplicate records:{df.duplicated().sum()}")

            #dropping duplicate records from dataset 
            logging.info(f"Dropping duplicate record")
            df.drop_duplicates(inplace=True)

            logging.info(f" Rechecking any duplicate record or not")
            logging.info(f"Duplicate record :{df.duplicated().sum()}")

           
            logging.info(f"Any missing value inside dataset{df.isna().sum().sort_values(ascending=False)}")
            #Save data in feature store
            logging.info("Create feature store folder if not available")
            #Create feature store folder if not available
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)



            logging.info("split dataset into train and test set")
            #split dataset into train and test set
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size)
            
            logging.info("create dataset directory folder if not available")
            #create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)
            
            

            #Prepare artifact
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise ShippingException(error_message=e, error_detail=sys)
            logging.ERROR(e)