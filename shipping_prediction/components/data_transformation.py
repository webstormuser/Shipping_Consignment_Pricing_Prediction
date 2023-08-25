from shipping_prediction.entity import artifact_entity, config_entity
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from typing import Optional
import os
import sys
import pandas as pd
import numpy as np
import pdb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from shipping_prediction import utils
from shipping_prediction.config import TARGET_COLUMN
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_validation_artifact:artifact_entity.DataValidationArtifact):
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
            cat_features =["PO_/_SO_#","ASN/DN_#","Country","Fulfill_Via","Vendor_INCO_Term","Shipment_Mode","Sub_Classification","First_Line_Designation"] # List of categorical feature column names
            num_features = ["Unit_of_Measure_(Per_Pack)","Line_Item_Quantity","Pack_Price","Unit_Price","Weight_(Kilograms)","Freight_Cost_(USD)","Line_Item_Insurance_(USD)","Days_to_Process"] # List of numerical feature column names
            # Create transformers for categorical and numerical features
            categorical_transformer = Pipeline(
                steps=[
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
                        ('model',TransformedTargetRegressor(regressor=xgb_model, transformer=target_scaler))
                    ])
            return final_pipeline
        except Exception as e:
            logging.error(f"An error occurred during data validation: {str(e)}")
            raise ShippingException(e,sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            
            #reading training and testing file
            logging.info(f"loading validated train and test file path")

            validated_train_file_path =self.data_validation_artifact.validated_train_file_path
            validated_test_file_path = self.data_validation_artifact.validated_test_file_path

            logging.info(f"Loading train and test df from validation")

            valid_train_df = pd.read_csv(validated_train_file_path)

            valid_test_df = pd.read_csv(validated_test_file_path)

            logging.info(f" Shape of Train df-->{valid_train_df.shape}")
            logging.info(f"Shape of test df -->{valid_test_df.shape}")

            logging.info(f"train df columns-->{valid_train_df.columns}")

            logging.info(f"test df columns--->{valid_test_df.columns}")
            
            #selecting input feature for train and test dataframe
            input_feature_train_df = valid_train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = valid_test_df.drop(TARGET_COLUMN,axis=1)

            logging.info(f"input feature train df and test df shape {input_feature_train_df.shape,input_feature_test_df.shape}")

            #selecting target feature for train and test dataframe
            target_feature_train_df = valid_train_df[TARGET_COLUMN]
            target_feature_test_df = valid_test_df[TARGET_COLUMN]

            logging.info(f"shape of target feature in train and test{target_feature_train_df.shape,target_feature_test_df.shape}")
            
            logging.info(f"Applying robustscale on target feature")
            logging.info(f"Applying robust scale on target feature")
            # Applying robust scaler on target feature

            target_scaler = RobustScaler()
            target_scaler.fit(target_feature_train_df.values.reshape(-1, 1))
            y_train_transformed = target_scaler.transform(target_feature_train_df.values.reshape(-1, 1))
            y_test_transformed = target_scaler.transform(target_feature_test_df.values.reshape(-1, 1))
            
            logging.info(f"Shape of y_train_transformed ==> {y_train_transformed.shape}")
            logging.info(f"Shape of y_test_transformed --> {y_test_transformed.shape}")


            # getting transformer object
            transformation_pipeline = DataTransformation.get_data_transformer_object()

            
            transformation_pipeline.fit(input_feature_train_df)

            #transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)

            
            # Combining input features and transformed target features
            train_arr = np.hstack((input_feature_train_arr, y_train_transformed))
            test_arr = np.hstack((input_feature_test_arr, y_test_transformed))
            
           #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

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
        
      
    
       