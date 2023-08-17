from shipping_prediction.logger import logging
from shipping_prediction.exception import ShippingException
from  shipping_prediction.utils import get_collection_as_dataframe
import os,sys
from  shipping_prediction.entity import config_entity
from shipping_prediction.entity import artifact_entity
from shipping_prediction.components.data_ingestion import DataIngestion
from shipping_prediction.components.data_validation import DataValidation
from shipping_prediction.components.data_transformation import DataTransformation
from shipping_prediction.pipeline.training_pipeline import start_training_pipeline

file_path="SCMS_Delivery_History_Dataset.csv"

if __name__=="__main__":
    try:
        start_training_pipeline()
    except Exception as e:
        print(e)