from shipping_prediction.predictor import ModelResolver 
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
import os,sys
from shipping_prediction.utils import load_object,save_object 
from shipping_prediction.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact
from shipping_prediction.entity.config_entity import ModelPusherConfig

class ModelPusher:
    def __init__(self,model_pusher_config:ModelPusherConfig,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config=model_pusher_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            logging.error(f"error occurred: {str(e)}")
            raise ShippingException(e,sys)
        
        
    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            #load object
            logging.info(f"Loading transformer model and target encoder")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)
          
            #model pusher dir
            logging.info(f"Saving model into model pusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)


            #saved model dir
            logging.info(f"Saving model in saved model dir")
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()
            
            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
            saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            logging.error(f"error occurred {str(e)}")
            raise ShippingException(e,sys)