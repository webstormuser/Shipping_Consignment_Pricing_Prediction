from shipping_prediction.entity import artifact_entity, config_entity
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from typing import Optional
import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from shipping_prediction.components.data_validation import DataValidation
from shipping_prediction import utils
from shipping_prediction.config import TARGET_COLUMN
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

class DataTransformation:
    def __init__(self,data_transformation_config: config_entity.DataTransformationConfig,
                data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                ):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise ShippingException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            # Define categorical and numerical features (replace with your actual feature lists)
            cat_features =['PO_/_SO_#','ASN/DN_#','Country','Fulfill_Via','Vendor_INCO_Term','Shipment_Mode','Sub_Classification','First_Line_Designation'] # List of categorical feature column names
            num_features = ['Unit_of_Measure_(Per_Pack)','Line_Item_Quantity','Pack_Price','Unit_Price','Weight_(Kilograms)','Freight_Cost_(USD)','Line_Item_Insurance_(USD)','Days_to_Process'] # List of numerical feature column names
            # Create transformers for categorical and numerical features
            categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
                            ])
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
            # Create an XGBoostRegressor
            xgb_model = XGBRegressor(n_estimators=100, max_depth=3)  # Modify parameters as needed

            #applying robust scaler on target feature
            target_scaler=RobustScaler()

            # Create a pipeline that includes preprocessing and XGBoost model
            final_pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', TransformedTargetRegressor(regressor=xgb_model, transformer=target_scaler))
                    ])
            return final_pipeline
        except Exception as e:
            raise ShippingException(e,sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            logging.info("Reading train and test dataframe")
            print("Reading train and test dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            print("checking any duplicate records inside train and tst")
            logging.info("Checking any duplicate records inside train and test dataframe")
            print(train_df.duplicated().sum(),test_df.duplicated().sum())
            logging.info(f"Train duplicates{train_df.duplicated().sum()}")
            logging.info(f"Test duplicates{test_df.duplicated().sum()}")

            logging.info("Dropping duplicates from train and test dataframe")
            print("dropping duplicate from train and test df")
            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)
            
            logging.info("Cross checking after dropping duplicates from train and test dataframe")
            logging.info(f"Train duplicate count{train_df.duplicated().sum()}")
            logging.info(f"Test duplicate count{test_df.duplicated().sum()}")
            print(f"after dropping{train_df.duplicated().sum(),test_df.duplicated().sum()}")
            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # getting transformer object
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # transforming input features
            logging.info("Transforming input features from train and test dataframe")
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

             # Converting target feature Series to numpy arrays
            target_feature_train_arr = target_feature_train_df.values
            target_feature_test_arr = target_feature_test_df.values


            # save numpy array
            logging.info("saving numpy array")
            utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_path, array=target_feature_train_arr
            )

            utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_path, array=target_feature_test_arr
            )

            logging.info("Saving transformer pipeline object ")
            utils.save_object(
                file_path=self.data_transformation_config.transform_object_path,
                obj=transformation_pipeline,
            )
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
            
            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ShippingException(e, sys)


       
    
       