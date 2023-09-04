from shipping_prediction.predictor import ModelResolver
from shipping_prediction.logger import logging
from shipping_prediction.exception import ShippingException
from shipping_prediction.utils import load_object
from shipping_prediction.entity import config_entity, artifact_entity
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from shipping_prediction.config import TARGET_COLUMN
import sys
import numpy as np

class ModelEvaluation:
    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_validation_artifact: artifact_entity.DataValidationArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact):

        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            logging.error(f"Error occurred {str(e)}")
            raise ShippingException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("If the saved model folder has a model, then we will compare which model is best trained or the model from the saved model folder.")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True,
                    is_accuracy_improved=None
                )
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Finding location of transformer model
            logging.info("Finding location of transformer model")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()

            # Previous trained objects
            logging.info(f"Loading previous trained model")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)

            # Currently trained model objects
            logging.info(f"Loading currently trained model")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)

            # Loading test df from validation
            logging.info("Loading test df for prediction")
            test_df = pd.read_csv(self.data_validation_artifact.validated_test_file_path)
            logging.info(f"Test df shape {test_df.shape}")
            target_df = test_df[TARGET_COLUMN]
            logging.info(f"Target_df shape {target_df.shape}")

            # Transforming input features using the previous trained transformer
            logging.info("Transforming input features using the previous trained transformer")
            input_feature_names = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_names])

            # Predicting and calculating R-squared score for the previous trained model
            logging.info("Predicting and calculating R-squared score for the previous trained model")
            y_pred = model.predict(input_arr)
            previous_model_r2 = r2_score(y_true=target_df, y_pred=y_pred)
            logging.info(f"R2 score using the previous trained model: {previous_model_r2}")

            # Transforming input features using the current trained transformer
            logging.info("Transforming input features using the current trained transformer")
            input_feature_names = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(test_df[input_feature_names])

            # Predicting and calculating R-squared score for the currently trained model
            logging.info("Predicting and calculating R-squared score for the currently trained model")
            y_pred = current_model.predict(input_arr)
            current_model_r2 = r2_score(y_true=target_df, y_pred=y_pred)
            logging.info(f"R2 score using the current trained model: {current_model_r2}")
            

            # Calculate the predictions
            y_pred_original = y_pred
            # Calculate and print the first 5 predictions
            print(f"First 5 Predictions: {np.exp(y_pred[:5]) - 1}")
            # Exponentiate each element and subtract 1, then slice the first 5 predictions
            
            # Calculating Mean Squared Error (MSE) for both models
            previous_model_mse = mean_squared_error(y_true=target_df, y_pred=y_pred)
            logging.info(f"Previous trained model MSE: {previous_model_mse}")
            current_model_mse = mean_squared_error(y_true=target_df, y_pred=y_pred)
            logging.info(f"Currently trained model MSE: {current_model_mse}")

            # Calculating Adjusted R-squared score for both models
            num_features = input_arr.shape[1]
            n = len(target_df)
            previous_model_adj_r2 = 1 - ((1 - previous_model_r2) * (n - 1) / (n - num_features - 1))
            logging.info(f"Previous model adjusted_r2_score is {previous_model_adj_r2}")
            current_model_adj_r2 = 1 - ((1 - current_model_r2) * (n - 1) / (n - num_features - 1))
            logging.info(f"Currently trained model adjusted r2_score is {current_model_adj_r2}")
            # Predictions using the trained model

            # Comparing model performances
            if current_model_adj_r2 <= previous_model_adj_r2:
                logging.info("Current trained model is not better than the previous model")
                raise Exception("Current trained model is not better than the previous model")

            # Creating and returning the model evaluation artifact
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                is_accuracy_improved=None,
                previous_model_r2=previous_model_r2,
                current_model_r2=current_model_r2,
                previous_model_mse=previous_model_mse,
                current_model_mse=current_model_mse,
                previous_model_adj_r2=previous_model_adj_r2,
                current_model_adj_r2=current_model_adj_r2,
                y_pred_original=y_pred_original  # Include original predictions
            )
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            raise ShippingException(e, sys)
