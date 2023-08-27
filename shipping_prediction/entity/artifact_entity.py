from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path:str
    validated_train_file_path: str # Add this attribute
    validated_test_file_path: str # Add this attribute

@dataclass
class DataTransformationArtifact:
    transform_object_path:str
    transformed_train_path:str
    transformed_test_path:str
    target_scaler_path:str

@dataclass    
class ModelTrainerArtifact:
    model_path:str 
    r2_score:float
    rmse_score:float
    adjusted_r2_score:float
    