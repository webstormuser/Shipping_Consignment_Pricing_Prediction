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
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

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
    def get_target_transformer_object(cls) -> Pipeline:
        try:
            TargetScaler = Pipeline(steps=[
                ('scaler', RobustScaler())
            ])
            return TargetScaler
        except Exception as e:
            logging.error(f"An error occurred during target transformation: {str(e)}")
            raise ShippingException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            # Define categorical and numerical features (replace with your actual feature lists)
            cat_features =['PO_SO','ASN_DN','Country','Fulfill_Via','Vendor_INCO_Term','Shipment_Mode','Sub_Classification','First_Line_Designation'] # List of categorical feature column names
            num_features = ['Unit_of_Measure_Per_Pack','Line_Item_Quantity','Pack_Price','Unit_Price','Weight_Kilograms','Freight_Cost_USD','Line_Item_Insurance_USD','Days_to_Process'] # List of numerical feature column names
            print(cat_features)
            print(num_features)
            print(TARGET_COLUMN)
            # Create transformers for categorical and numerical features
            logging.info(f"applying categorical transformer")
            categorical_transformer = Pipeline(
                        steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
                            ])
            logging.info(f"applying numerical transformer")
            numerical_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', RobustScaler())
                        ])
           # Combine transformers using ColumnTransformer
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
            raise ShippingException(e,sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
           
            logging.info(f"loading validated train and test file path")
            validated_train_file_path =self.data_validation_artifact.validated_train_file_path
            validated_test_file_path = self.data_validation_artifact.validated_test_file_path

            logging.info(f"Loading train and test df from validation")
            valid_train_df = pd.read_csv(validated_train_file_path)
            valid_test_df = pd.read_csv(validated_test_file_path)

            logging.info(f" Shape of Train df-->{valid_train_df.shape}")
            logging.info(f"Shape of test df -->{valid_test_df.shape}")

            logging.info(f"train df columns-->{valid_train_df.columns.to_list()}")
            logging.info(f"test df columns--->{valid_test_df.columns.to_list()}")
            
            #selecting input feature for train and test data frame
            input_feature_train_df = valid_train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = valid_test_df.drop(TARGET_COLUMN,axis=1)

            logging.info(f"input feature train df and test df shape {input_feature_train_df.shape,input_feature_test_df.shape}")
            logging.info(f"input_feature_train_df_columns{input_feature_train_df.columns.to_list()}")
            logging.info(f"input_feature_test_df_columns{input_feature_test_df.columns.to_list()}")
           
            #selecting target feature for train and test data frame
            target_feature_train_df = valid_train_df[TARGET_COLUMN]
            target_feature_test_df = valid_test_df[TARGET_COLUMN]

            logging.info(f"shape of target feature in train and test{target_feature_train_df.shape,target_feature_test_df.shape}")
            logging.info(f"target_feature_train_df{target_feature_train_df}")
            logging.info(f"target_feature_test_df{target_feature_test_df}")
            logging.info(f"Before applying scaler shape of target feature{target_feature_train_df.shape,target_feature_test_df.shape}")
        
            # Reshape target features
            target_feature_train_arr = target_feature_train_df.values.reshape(-1, 1)
            target_feature_test_arr = target_feature_test_df.values.reshape(-1, 1)
            logging.info(f"After  reshaping and before applying scaler {target_feature_train_arr.shape,target_feature_test_arr.shape}")
          
            TargetScaler =  DataTransformation.get_target_transformer_object()
            TargetScaler.fit(target_feature_train_arr)
            # Transform target features
            target_feature_train_scaled = TargetScaler.transform(target_feature_train_arr)
            target_feature_test_scaled = TargetScaler.transform(target_feature_test_arr)
            logging.info(f"after applying scaler shape of target is {target_feature_train_scaled.shape,target_feature_test_scaled.shape}")
            
            logging.info(f"After applying scaler and reshaping back to original ")       
            # Reshape the target features to match the number of rows in input features
            target_feature_train_scaled_reshaped = target_feature_train_scaled.reshape(-1,1)
            target_feature_test_scaled_reshaped = target_feature_test_scaled.reshape(-1,1)
            
            logging.info(f"shape of target scaler reshaped {target_feature_train_scaled_reshaped.shape,target_feature_test_scaled_reshaped.shape}")
            
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

            logging.info(f"shape of input feature train and test arr{input_feature_train_arr.shape,input_feature_test_arr.shape}")
            
           # Printing shapes of arrays for debugging
            print("Shapes before concatenation:")
            print("input_feature_train_arr-->",input_feature_train_arr.shape)
            print("input_feature_test_arr-->",input_feature_test_arr.shape)
            print("input_feature_train_arr_datatype-->",input_feature_train_arr.dtype)
            print("input_feature_train_arr_datatype-->",input_feature_test_arr.dtype)
            print("target_feature_train_scaled_reshaped_shape",target_feature_train_scaled_reshaped.shape)
            print("target_feature_test_scaled_reshaped",target_feature_test_scaled_reshaped.shape)
            print("type of input_feature_train_arr-->",type(input_feature_train_arr))
            print("type of input_feature_test_arr-->",type(input_feature_test_arr))
            print("type of target_feature_train_scaled_reshaped",type(target_feature_train_scaled_reshaped))
            print("type of target_feature_test_scaled_reshaped",type(target_feature_test_scaled_reshaped))
            # Convert sparse matrices to dense numpy arrays
            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()
            print("after to array type of iput feature train test--->",type(input_feature_train_arr),type(input_feature_test_arr))
            # Concatenating input features with target features
            logging.info("Concatenating input feature with target feature in train and test")
            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_scaled_reshaped), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_scaled_reshaped), axis=1)

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
            
            logging.info(f"Saving Target scaler object")
            utils.save_object(
                file_path=self.data_transformation_config.target_scaler_path,
                obj=TargetScaler,
            )

            data_transformation_artifact=artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_scaler_path=self.data_transformation_config.target_scaler_path,
            )
            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            logging.error(f"An error occurred during data validation: {str(e)}")
            raise ShippingException(e, sys)
        
      
    
       
       