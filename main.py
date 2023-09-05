from shipping_prediction.logger import logging
from shipping_prediction.exception import ShippingException
from  shipping_prediction.utils import get_collection_as_dataframe
import os,sys
from  shipping_prediction.entity import config_entity
from shipping_prediction.entity import artifact_entity
from shipping_prediction.components.data_ingestion import DataIngestion
from shipping_prediction.components.data_validation import DataValidation
from shipping_prediction.components.data_transformation import DataTransformation
from shipping_prediction.components.model_trainer import ModelTrainer
from shipping_prediction.components.model_evaluation import ModelEvaluation
from shipping_prediction.components.model_pusher import ModelPusher 
from shipping_prediction.pipeline.training_pipeline import start_training_pipeline
from shipping_prediction.pipeline.batch_prediction import start_batch_prediction
from shipping_prediction.predictor import ModelResolver
file_path="clean_SCMS_Delivery_History_Dataset.csv"



if __name__=="__main__":
    try:
        start_training_pipeline()
        output_file=start_batch_prediction(file_path)
        print(output_file)
    except Exception as e:
        print(e)