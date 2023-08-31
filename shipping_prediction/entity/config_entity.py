import os,sys
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from datetime import datetime
from shipping_prediction.config import TARGET_COLUMN
import yaml

FILE_NAME = "clean_SCMS_Delivery_History_Dataset.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_SCALER_OBJECT_FILE_NAME = "TargetScaler.pkl"
MODEL_FILE_NAME = "model.pkl"


class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise ShippingException(e,sys)     


class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name="clean_Logistic1"
            self.collection_name="clean_shipping_data1"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception  as e:
            raise ShippingException(e,sys)   

    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise ShippingException(e,sys) 

class DataValidationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_validation")
        self.report_file_path=os.path.join(self.data_validation_dir, "report.yaml")
        self.unrelevant_columns:list = ['ID','Project_Code','PQ_#','Managed_By','PO_Sent_to_Vendor_Date','Product_Group','Scheduled_Delivery_Date','Delivered_to_Client_Date','Delivery_Recorded_Date','Vendor','Item_Description','Molecule/Test_Type','Brand','Dosage','Dosage_Form','Manufacturing_Site','PQ_First_Sent_to_Client_Date']
        self.base_file_path = os.path.join("clean_SCMS_Delivery_History_Dataset.csv")
        self.validated_train_file_path = os.path.join(self.data_validation_dir, "validated_train.csv")
        self.validated_test_file_path = os.path.join(self.data_validation_dir, "validated_test.csv")

class DataTransformationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir=os.path.join(training_pipeline_config.artifact_dir,"data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path =  os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv","npz"))
        self.transformed_test_path =os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv","npz"))
        self.target_scaler_path = os.path.join(self.data_transformation_dir,"TargetScaler",TARGET_SCALER_OBJECT_FILE_NAME)
class ModelTrainerConfig:
    
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir=os.path.join(training_pipeline_config.artifact_dir,"model_trainer")
        self.model_path=os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
        self.expected_score=0.8
        self.overfitting_threshold=0.1
                
class ModelEvaluationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold=0.01
        
class ModelPusherConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir , "model_pusher")
        self.saved_model_dir = os.path.join("saved_models")
        self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir,TRANSFORMER_OBJECT_FILE_NAME)
        self.pusher_target_scaler_path = os.path.join(self.pusher_model_dir,TARGET_SCALER_OBJECT_FILE_NAME)
        


    
                