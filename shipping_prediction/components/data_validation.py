from shipping_prediction.entity import artifact_entity,config_entity
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency
from typing import Optional
import os,sys 
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
            # Convert column names in the list to lowercase with underscores
            column_list_lower_underscore = [col.lower().replace(' ', '_') for col in unrelevant_columns]

            # Get the matching column names from the DataFrame
            matching_columns = [col for col in df.columns if col.lower() in column_list_lower_underscore]
            #droppping unrelevant columns which are not usefull for model bulding 
            logging.info(f" Columns to drop :{unrelevant_columns}")
            df.drop(matching_columns,axis=1,inplace=True)
            #return None no columns left
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise ShippingException(e, sys)



    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
           
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base} is not available.]")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False
            return True
        except Exception as e:
            raise ShippingException(e, sys)
        
        
        
    def check_data_drift_categorical(self, base_data: pd.Series, current_data: pd.Series):
    # Create frequency tables for the feature in both datasets
        table1 = pd.value_counts(base_data).sort_index()
        table2 = pd.value_counts(current_data).sort_index()

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

        # Checking data drift for numerical features
            for base_column in base_columns_num:
                base_data, current_data = base_df[base_column], current_df[base_column]
            # Null hypothesis is that both column data drawn from the same distribution

            logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype}")
            same_distribution = ks_2samp(base_data, current_data)

            if same_distribution.pvalue > 0.05:
                # We are accepting the null hypothesis
                drift_report[base_column] = {
                    "pvalues": float(same_distribution.pvalue),
                    "same_distribution": True
                }
            else:
                drift_report[base_column] = {
                    "pvalues": float(same_distribution.pvalue),
                    "same_distribution": False
                }
                # Different distribution

        # Checking data drift for categorical features
            for col in categorical_cols_base_df:
                base_data, current_data = base_df[col], current_df[col]
                logging.info(f"Hypothesis {col}: {base_data.dtype}, {current_data.dtype}")
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
            raise ShippingException(e, sys)

    
    
    
    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:

            unrelevant_columns=self.data_validation_config.unrelevant_columns
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            # Replace spaces with underscores in the base DataFrame's column names
            base_df.columns = [col.replace(' ', '_') for col in base_df.columns]
            #base_df has na as null
           

            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
  
      
            logging.info(f"Adding new columns from data ingestion to base_df if missing")
            new_columns_added = set(train_df.columns) - set(base_df.columns)
            for new_col in new_columns_added:
            # Check if the new column already exists in base_df using both original and underscore-replaced names
                if new_col not in base_df.columns and new_col.replace('_', ' ') not in base_df.columns:
                    base_df[new_col] = np.nan  # Adding the new column with NaN values
            logging.info(f" base_df{base_df.columns}")

            logging.info(f"train_df{train_df.columns}")

     
            logging.info(f"test_df{test_df.columns}")
            
       
            logging.info(f"Dropping unrelevent columns from base df")
            base_df=self.drop_unrelevant_columns(df=base_df,column_list=unrelevant_columns,report_key_name="dropping_unrelevent_columns_from_base_df")
            
            logging.info(f" Dropping unrelevent columns from train_df")
            train_df=self.drop_unrelevant_columns(df=train_df,column_list=unrelevant_columns,report_key_name="dropping_unrelevent_columns_frombase_df")

            logging.info(f" Dropping unrelevent columns from test_df")
            test_df=self.drop_unrelevant_columns(df=test_df,column_list=unrelevant_columns,report_key_name="dropping_unrelevent_columns_frombase_df")

            


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


            #write the report
            logging.info("Write reprt in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
                                data=self.validation_error)
            logging.info(f"{base_df.columns}")
            logging.info(f"{train_df.columns}")
            logging.info(f"{test_df.columns}")

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path,)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ShippingException(e,sys)