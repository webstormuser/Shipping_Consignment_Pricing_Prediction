from shipping_prediction.predictor import ModelResolver
from shipping_prediction.logger import logging
from shipping_prediction.exception import ShippingException
from shipping_prediction.utils import load_object
from shipping_prediction.entity import config_entity, artifact_entity
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from shipping_prediction.config import TARGET_COLUMN
import sys

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
            logging.info("if saved model folder has model the we will compare " "which model is best trained or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True,
                    is_accuracy_improved=None
                )
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Finding location of transformer model and target scaler
            logging.info("Finding location of transformer model and target scaler")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_scaler_path = self.model_resolver.get_latest_target_scaler_path()

            # Previous trained objects
            logging.info(f"loading previous trained model and target_scaler")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            TargetScaler  = load_object(file_path=target_scaler_path)

            # Currently trained model objects
            logging.info(f"Loafing currently trained model and target_scaler")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_scaler = load_object(file_path=self.data_transformation_artifact.target_scaler_path)

            # Loading test df from validation
            logging.info("Loading test df for prediction")
            test_df = pd.read_csv(self.data_validation_artifact.validated_test_file_path)
            logging.info(f"test df shape{test_df.shape} ")
            target_df = test_df[TARGET_COLUMN]
            logging.info(f"target_df shape--{target_df.shape}")
            # Transforming the target feature
            y_true = current_target_scaler.transform(target_df.values.reshape(-1, 1))

            # Calculating R-squared score using the previous trained model
            logging.info("Loading input feature names")
            input_feature_name = list(transformer.feature_names_in_)
            logging.info(f"Input feature names: {input_feature_name}")
            logging.info("Loading transformer to transform input features")
            input_arr = transformer.transform(test_df[input_feature_name])
            logging.info("Predicting and calculating r2 score for previous trained model")
            y_pred = model.predict(input_arr)
            previous_model_r2 = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"R2  score using the previous trained model: {previous_model_r2}")

            # Calculating R-squared score using the current trained model
            input_feature_name = list(current_transformer.feature_names_in_)
            logging.info(f"Input feature names from current trained model: {input_feature_name}")
            input_arr = current_transformer.transform(test_df[input_feature_name])
            logging.info("Predicting and calculating r2 score for currently trained model")
            y_pred = current_model.predict(input_arr)
            y_true = current_target_scaler.transform(target_df.values.reshape(-1, 1)).flatten()  # Flatten y_true
            y_pred =  y_pred.reshape(-1, 1)  # Reshape y_pred to a 2D array
            current_model_r2 = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"R2 score using the current trained model: {current_model_r2}")
            #print(f"Prediction using the trained model: {current_target_scaler.inverse_transform(y_pred[:5])}")
            print("Prediction using the trained model:", ' '.join(map(str, [int(round(value[0])) for value in current_target_scaler.inverse_transform(y_pred[:5])])))
            # Calculating Mean Squared Error (MSE)
            previous_model_mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info(f"Previous trained model MSE: {previous_model_mse}")
            current_model_mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info(f"Currently trained model MSE: {current_model_mse}")

            # Calculating Adjusted R-squared score
            num_features = input_arr.shape[1]
            n = len(y_true)
            previous_model_adj_r2 = 1 - ((1 - previous_model_r2) * (n - 1) / (n - num_features - 1))
            logging.info(f"Previous model adjusted_r2_score is {previous_model_adj_r2}")
            current_model_adj_r2 = 1 - ((1 - current_model_r2) * (n - 1) / (n - num_features - 1))
            logging.info(f"Currently trained model adjusted r2_score is {current_model_adj_r2}")

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
                current_model_adj_r2=current_model_adj_r2
            )
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
                logging.error(f"Error occurred: {str(e)}")
                raise ShippingException(e, sys)
