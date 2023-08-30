import os 
from shipping_prediction.entity.config_entity import TRANSFORMER_OBJECT_FILE_NAME,TARGET_SCALER_OBJECT_FILE_NAME,MODEL_FILE_NAME
from glob import glob
from typing import Optional
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
import sys 

class ModelResolver:
    
    def __init__(self,model_registry:str = "saved_models",
                transformer_dir_name="transformer",
                target_scaler_dir_name = "TargetScaler",
                model_dir_name = "model"):

        self.model_registry=model_registry
        os.makedirs(self.model_registry,exist_ok=True)
        self.transformer_dir_name = transformer_dir_name
        self.target_scaler_dir_name=target_scaler_dir_name
        self.model_dir_name=model_dir_name
                
    def get_latest_dir_path(self)->Optional[str]:
        try:
            dir_names = os.listdir(self.model_registry)
            logging.info(f"dir _name:{dir_names}")
            if len(dir_names)==0:
                return None
            dir_names = list(map(int,dir_names))
            latest_dir_name = max(dir_names)
            logging.info(f"latest dir names:{latest_dir_names}")
            return os.path.join(self.model_registry,f"{latest_dir_name}")
        except Exception as e:
            logging.info(f" Error occurred :{str(e)}")
            raise ShippingException(e,sys)
        
    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            logging.error(f" Error occurred during  getting latest dir path: {str(e)}")
            raise ShippingException(e,sys)
        
    def get_latest_transformer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not available")
            return os.path.join(latest_dir,self.transformer_dir_name,TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            logging.error(f" Error occurred during  getting latest dir path: {str(e)}")
            raise ShippingException(e,sys)
        
    def get_latest_target_scaler_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Target scaler object  is not available")
            return os.path.join(latest_dir,self.target_scaler_dir_name,TARGET_SCALER_OBJECT_FILE_NAME)
        except Exception as e:
            logging.error(f" Error occurred during  getting latest dir path: {str(e)}")
            raise ShippingException(e,sys)
        
        
    def get_latest_save_dir_path(self)->str:
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir==None:
                return os.path.join(self.model_registry,f"{0}")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry,f"{latest_dir_num+1}")
        except Exception as e:
            logging.error(f" Error occurred during  getting latest dir path: {str(e)}")
            raise ShippingException(e,sys)

    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            logging.error(f" Error occurred during  getting latest dir path: {str(e)}")
            raise ShippingException(e,sys)

    def get_latest_save_transformer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.transformer_dir_name,TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            logging.error(f" Error occurred during  getting latest dir path: {str(e)}")
            raise ShippingException(e,sys)
        
    def get_latest_save_target_scaler_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.target_scaler_dir_name,TARGET_SCALER_OBJECT_FILE_NAME)
        except Exception as e:
            logging.error(f" Error occurred during  getting latest dir path: {str(e)}")
            raise ShippingException(e,sys)