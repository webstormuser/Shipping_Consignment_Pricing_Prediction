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
    r2_score_train:float
    r2_score_test:float
    adjusted_r2_score_train:float
    adjusted_r2_score_test:float
    mse_train:float
    mse_test:float
    
@dataclass 
class ModelEvaluationArtifact:
    is_model_accepted:bool
    is_accuracy_improved:bool

    
@dataclass
class ModelPusherArtifact:
    pusher_model_dir:str 
    saved_model_dir:str