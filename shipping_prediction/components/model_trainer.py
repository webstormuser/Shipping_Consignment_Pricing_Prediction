from shipping_prediction.entity import artifact_entity,config_entity
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from shipping_prediction import utils
import os,sys
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise ShippingException(e, sys)

    def train_model(self,x,y):
        try:
            clf=XGBRegressor(n_estimators=100, max_depth=3)
            clf.fit(x,y)
            return clf
        except Exception as e:
            logging.error(f"An error occurred during data validation: {str(e)}")
            raise ShippingException(e,sys)

    def initiate_model_training(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"loading train and test numpy array for model training ")  
            # Load the .npz file
            train_arr = utils.load_numpy_array_data(self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(self.data_transformation_artifact.transformed_test_path)
            logging.info(f"Loaded train arr shape: {train_arr.shape}")
            logging.info(f"Loaded test arr shape: {test_arr.shape}")
      
            # Splitting input and target feature from both train and test arr.
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            logging.info(f"Shape of X_train, X_test ---> {x_train.shape}, {x_test.shape}")
                
            logging.info(f"Train the model")
            model = self.train_model(x=x_train,y=y_train)
                
            logging.info(f"Calculating r2-score train")
            y_train_pred = model.predict(x_train)
            r2_score_train = r2_score(y_train,y_train_pred)
                
            logging.info(f"Calculating r2_score test")
            y_test_pred = model.predict(x_test)
            r2_score_test = r2_score(y_test,y_test_pred)
                
            logging.info(f"Train and test r2 score--->{r2_score_train,r2_score_test}")                
                
            logging.info(f"Calculating MSE train score")
            mse_train = mean_squared_error(y_train,y_train_pred)
                
            logging.info(f"Calculating MSE test score")
            mse_test = mean_squared_error(y_test,y_test_pred)
                
            logging.info(f"Train and test MSE score ---->{mse_train,mse_test}")
                
            # Calculate the number of predictor variables (columns) in x_train and x_test
            p_train = x_train.shape[1]
            p_test = x_test.shape[1]

            # Calculate Adjusted R2 scores for train and test datasets
            adjusted_r2_score_train = 1 - (1 - r2_score_train) * (len(y_train) - 1) / (len(y_train) - p_train - 1)
            adjusted_r2_score_test = 1 - (1 - r2_score_test) * (len(y_test) - 1) / (len(y_test) - p_test - 1)

            # Log the Adjusted R2 scores for train and test datasets
            logging.info(f"Adjusted R2 score for train dataset: {adjusted_r2_score_train:.4f}")
            logging.info(f"Adjusted R2 score for test dataset: {adjusted_r2_score_test:.4f}")  
                
            logging.info(f"Checking if our model is underfitting or not")
            if adjusted_r2_score_test < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {adjusted_r2_score_test}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(adjusted_r2_score_train-adjusted_r2_score_test)
            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            logging.info(f"Saving trained model")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)
                
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(
            model_path=self.model_trainer_config.model_path, 
            r2_score_train = r2_score_train,
            r2_score_test = r2_score_test,
            mse_train = mse_train,
            mse_test=mse_test,
            adjusted_r2_score_train = adjusted_r2_score_train,
            adjusted_r2_score_test = adjusted_r2_score_test )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact         
        except Exception as e:
            logging.error(f"An error occurred during data validation: {str(e)}")
            raise ShippingException(e,sys)