from shipping_prediction.logger import logging
from shipping_prediction.exception import ShippingException
from  shipping_prediction.utils import get_collection_as_dataframe
import os,sys
from  shipping_prediction.entity import config_entity
from shipping_prediction.components.data_ingestion import DataIngestion
from shipping_prediction.components.data_validation import DataValidation
from shipping_prediction.components.data_transformation import DataTransformation
from shipping_prediction.components.model_trainer import ModelTrainer
from shipping_prediction.config import TARGET_COLUMN
import traceback

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

        #data transformation
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
                                                data_validation_artifact=data_validation_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        
        #model trainer
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_training()
        
    except Exception as e :
        logging.error(f"An error occurred during data validation: {str(e)}")
        raise ShippingException(e,sys)
    
     



       