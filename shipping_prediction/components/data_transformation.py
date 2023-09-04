# Import necessary libraries
from shipping_prediction.entity import artifact_entity, config_entity
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from shipping_prediction.config import TARGET_COLUMN
from shipping_prediction import utils
import warnings
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_validation_artifact: artifact_entity.DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise ShippingException(e, sys)
    
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            # Define categorical and numerical features (replace with your actual feature lists)
            cat_features = ['PO_SO', 'ASN_DN', 'Sub_Classification']
            num_features = ['Line_Item_Quantity', 'Pack_Price', 'Unit_Price', 'Weight_Kilograms', 'Freight_Cost_USD']
            
            # Create transformers for categorical and numerical features
            logging.info(f"Applying categorical transformer")
            categorical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
                ])
            logging.info(f"Applying numerical transformer")
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            
            # Combine transformers using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, cat_features),
                    ('num', numerical_transformer, num_features)
                ])
            
            # Create the final pipeline
            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
            ])
            return final_pipeline
        except Exception as e:
            logging.error(f"An error occurred during data validation: {str(e)}")
            raise ShippingException(e, sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            logging.info(f"Loading validated train and test file path")
            validated_train_file_path = self.data_validation_artifact.validated_train_file_path
            validated_test_file_path = self.data_validation_artifact.validated_test_file_path

            logging.info(f"Loading train and test dataframes from validation")
            valid_train_df = pd.read_csv(validated_train_file_path)
            valid_test_df = pd.read_csv(validated_test_file_path)

            logging.info(f"Shape of Train df: {valid_train_df.shape}")
            logging.info(f"Shape of Test df: {valid_test_df.shape}")

            logging.info(f"Train df columns: {valid_train_df.columns.to_list()}")
            logging.info(f"Test df columns: {valid_test_df.columns.to_list()}")
            
            # Selecting input features for train and test dataframes
            input_feature_train_df = valid_train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = valid_test_df.drop(TARGET_COLUMN, axis=1)

            logging.info(f"Input feature train df and test df shape: {input_feature_train_df.shape, input_feature_test_df.shape}")
            logging.info(f"Input_feature_train_df_columns: {input_feature_train_df.columns.to_list()}")
            logging.info(f"Input_feature_test_df_columns: {input_feature_test_df.columns.to_list()}")
           
            # Selecting target features for train and test dataframes
            target_feature_train_df = valid_train_df[TARGET_COLUMN]
            target_feature_test_df = valid_test_df[TARGET_COLUMN]

            logging.info(f"Shape of target feature in train and test: {target_feature_train_df.shape, target_feature_test_df.shape}")
            logging.info(f"Before applying log transformation on target features")
            
            # Applying log transformation to target features
            target_feature_train_df = np.log(target_feature_train_df + 1)
            target_feature_test_df = np.log(target_feature_test_df + 1)
           
            # Reshape target features
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)
            logging.info(f"After reshaping: {target_feature_train_arr.shape, target_feature_test_arr.shape}")
          
            # Getting transformation object
            logging.info("Getting transformation object from pipeline")
            transformation_pipeline = DataTransformation.get_data_transformer_object()

            # Fitting transformers
            logging.info("Fitting transformers on train data")
            transformation_pipeline.fit(input_feature_train_df)

            # Transforming train and test data
            logging.info("Transforming train and test data using fitted transformers")
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            logging.info(f"Shape of input feature train and test arr: {input_feature_train_arr.shape, input_feature_test_arr.shape}")

            # Concatenating input features with target features
            logging.info("Concatenating input feature with target feature in train and test")
            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)
            # Printing shapes of concatenated arrays
            print("Shapes after concatenation:")
            print("train_arr:", train_arr.shape)
            print("test_arr:", test_arr.shape)

            logging.info(f"shape of combined input and target feature in train and test{train_arr.shape,test_arr.shape}")
            
           #save numpy array
            logging.info(f"Saving transformed train array data")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            logging.info(f"Saving transformed test array data")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            logging.info(f"Saving transformer object ")
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
            obj=transformation_pipeline)
           
            data_transformation_artifact=artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path
            )
            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            logging.error(f"An error occurred during data validation: {str(e)}")
            raise ShippingException(e, sys)
