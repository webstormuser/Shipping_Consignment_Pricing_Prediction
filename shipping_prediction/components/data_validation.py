from shipping_prediction.entity import artifact_entity,config_entity
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
from typing import Optional
import os
import sys
import pandas as pd
from shipping_prediction import utils
import numpy as np
from shipping_prediction.config import TARGET_COLUMN



class DataValidation:


    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise ShippingException(e, sys)

    
    def drop_unrelevant_columns(self,df:pd.DataFrame,column_list:list,report_key_name:str)->Optional[pd.DataFrame]:
        '''This function drops the unrelevant columns from the dataset both from train and test '''
        try:
            unrelevant_columns=self.data_validation_config.unrelevant_columns
            #droppping unrelevant columns which are not usefull for model bulding 
            logging.info(f" Columns to drop :{unrelevant_columns}")
            df.drop(unrelevant_columns,axis=1,inplace=True)
            #return None no columns left
            if len(df.columns)==0:
                return None
            logging.info(f"After dropping columns from df shape of df{df.shape}")
            return df
        except Exception as e:
            raise ShippingException(e, sys)



    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
           
            base_columns = base_df.columns
            logging.info(f"base columns{base_columns}")
            current_columns = current_df.columns
            logging.info(f"current_columns{current_columns}")

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available.]")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False
            return True
        except Exception as e:
            raise ShippingException(e, sys)
        
        
        
    def check_data_drift_categorical(self, base_data, current_data):
        """
        Checks data drift for a categorical feature using the chi-square test.

        Args:
            base_data (pd.Series): First dataset containing the feature.
            current_data (pd.Series): Second dataset containing the feature.

        Returns:
            drift (bool): True if there is significant data drift, False otherwise.
            p_value (float): The p-value of the chi-square test.
        """
        # Create frequency tables for the feature in both datasets
        table1 = base_data.value_counts()
        table2 = current_data.value_counts()

        # Combine the two tables into a single table
        all_values = set(table1.index).union(set(table2.index))
        combined_table = {val: [0, 0] for val in all_values}

        for val, count in table1.items():
            combined_table[val][0] = count

        for val, count in table2.items():
            combined_table[val][1] = count

        # Convert the combined table into an array
        contingency_table = np.array(list(combined_table.values())).T

        # Perform chi-square test
        _, p_value, _, _ = chi2_contingency(contingency_table)

        # Determine if there is data drift based on the p-value
        drift = p_value < 0.05

        return drift, p_value

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str):
        try:
            drift_report = {}
            categorical_cols_base_df = base_df.select_dtypes(include=['object']).columns
            categorical_cols_current_df = current_df.select_dtypes(include=['object']).columns
            base_columns_num = base_df.select_dtypes(include=['number']).columns
            current_columns_num = current_df.select_dtypes(include=['number']).columns

            # Checking data drift for numerical Features
            for base_column in base_columns_num:
                base_data, current_data = base_df[base_column], current_df[base_column]
                # Null hypothesis is that both column data drawn from the same distribution
                logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype} ")
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    # We are accepting the null hypothesis
                    drift_report[base_column] = {
                        "p_values": float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                else:
                    drift_report[base_column] = {
                        "p_values": float(same_distribution.pvalue),
                        "same_distribution": False
                    }

            # Checking data drift for categorical features
            for col in categorical_cols_base_df:
                base_data, current_data = base_df[col], current_df[col]
                logging.info(f"Hypothesis {col}: {base_data.dtype}, {current_data.dtype} ")
                drift, p_value = self.check_data_drift_categorical(base_data, current_data)

                if drift:
                    drift_report[col] = {
                        "p_values": float(p_value),
                        "same_distribution": False
                    }
                else:
                    drift_report[col] = {
                        "p_values": float(p_value),
                        "same_distribution": True
                    }

            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise ShippingException(e,sys)
    
    
    
    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            unrelevant_columns=self.data_validation_config.unrelevant_columns

            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            logging.info(f"base_df shape-->{base_df.shape}")

            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"train_df shape-->{train_df.shape}")

            logging.info(f"Reading test data frame")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"test_df shape-->{test_df.shape}")
            
            logging.info(f"Any duplicate records inside base_df ,train_df,test_df{base_df.duplicated().sum(),train_df.duplicated().sum(),test_df.duplicated().sum()}")
            
            
            logging.info(f"Dropping duplicate from base_df ,train_df,test_df")
            base_df.drop_duplicates(inplace=True)
            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)
            
            logging.info(f"After dropping duplicate ")
            logging.info(f"Duplicate count in base_df,train_df,test_df-->{base_df.duplicated().sum(),train_df.duplicated().sum(),test_df.duplicated().sum()}")            
            base_df=self.drop_unrelevant_columns(df=base_df,column_list=unrelevant_columns,report_key_name="dropping_unrelevent_columns_from_base_df")
            
    
            train_df=self.drop_unrelevant_columns(df=train_df,column_list=unrelevant_columns,report_key_name="dropping_unrelevent_columns_from train_df")

            test_df=self.drop_unrelevant_columns(df=test_df,column_list=unrelevant_columns,report_key_name="dropping_unrelevent_columns_from test_df")

            # Add more logging messages here to understand what's happening
            logging.info(f"Train dataframe shape after dropping columns: {train_df.shape}")
            logging.info(f"Test dataframe shape after dropping columns: {test_df.shape}")

            logging.info(f"Is all required columns present in train df")
            train_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df,report_key_name="missing_columns_within_train_dataset")
            logging.info(f"Is all required columns present in test df")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df,report_key_name="missing_columns_within_test_dataset")


            if train_df_columns_status:
                logging.info(f"As all column are available in train df hence detecting data drift")
            self.data_drift(base_df=base_df, current_df=train_df,report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:
                logging.info(f"As all column are available in test df hence detecting data drift")
            self.data_drift(base_df=base_df, current_df=test_df,report_key_name="data_drift_within_test_dataset")

             # Create the directory if it doesn't exist
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)

            logging.info(f"Saving validated train and test dataframe")
            # Save the validated train and test dataframes to their respective paths
            train_df.to_csv(path_or_buf=self.data_validation_config.validated_train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_validation_config.validated_test_file_path, index=False, header=True)

           
            #write the report
            logging.info("Write reprt in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
                                data=self.validation_error)
           
            # Create the DataValidationArtifact object with updated attributes
            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path,
                validated_train_file_path=self.data_validation_config.validated_train_file_path,
                validated_test_file_path=self.data_validation_config.validated_test_file_path
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            logging.error(f"An error occurred during data validation: {str(e)}")
            raise ShippingException(e,sys)