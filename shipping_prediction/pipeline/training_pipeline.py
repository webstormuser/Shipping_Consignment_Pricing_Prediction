from shipping_prediction.logger import logging
from shipping_prediction.exception import CustomException
from  shipping_prediction.utils import get_collection_as_dataframe
import os,sys
from  shipping_prediction.entity import config_entity
from shipping_prediction.components.data_ingestion import DataIngestion
from shipping_prediction.components.data_validation import DataValidation

def start_training_pipeline():
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        #data ingestion
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        #data validation
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                        data_ingestion_artifact=data_ingestion_artifact)

        data_validation_artifact = data_validation.initiate_data_validation()
    except Exception as e :
        raise CustomException(e,sys)


        