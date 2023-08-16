import os,sys
from shipping_prediction.exception import CustomException
from shipping_prediction.logger import logging
from datetime import datetime
from shipping_prediction.config import TARGET_COLUMN

FILE_NAME = "SCMS_Delivery_History_Dataset.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"


class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise CustomException(e,sys)     


class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name="Logistic"
            self.collection_name="shipping_data"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception  as e:
            raise CustomException(e,sys)   

    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise CustomException(e,sys) 

class DataValidationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_validation")
        self.report_file_path=os.path.join(self.data_validation_dir, "report.yaml")
        self.threshold=0.3
        self.unrelevant_columns:list =['ID','Project_Code','PQ_#','Item_Description','Managed_By','PO_Sent_to_Vendor_Date','Product_Group','Molecule/Test_Type','Brand','Dosage_Form','Dosage','Manufacturing_Site','Vendor','PQ_First_Sent_to_Client_Date','Scheduled_Delivery_Date','Delivered_to_Client_Date','Delivery_Recorded_Date']
        self.base_file_path = os.path.join("SCMS_Delivery_History_Dataset.csv")
