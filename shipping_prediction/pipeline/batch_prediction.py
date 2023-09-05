import pandas as pd
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
import os, sys
from shipping_prediction.predictor import ModelResolver
from shipping_prediction.utils import load_object
from datetime import datetime
import numpy as np

PREDICTION_DIR = "prediction"

def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file: {input_file_path}")
        df = pd.read_csv(input_file_path)

        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())

        input_feature_names = list(transformer.feature_names_in_)
        logging.info(f"Input feature names: {input_feature_names}")

        input_arr = transformer.transform(df[input_feature_names])
        logging.info(f"Shape of input_arr before reshaping: {input_arr.shape}")

        # Print out the first few rows of the input_arr for inspection
        logging.info(f"First few rows of input_arr: {input_arr[:5]}")

        # Reshape the input array to match the model's expected input shape
        num_features = input_arr.shape[1]
        logging.info(f"num_features: {num_features}")
        input_arr_reshaped = input_arr.reshape(-1, num_features)
        logging.info(f"input_arr_reshaped_shape: {input_arr_reshaped.shape}")

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())

        # Debug print to check the shape of input_arr_reshaped before prediction
        logging.info(f"Shape of input_arr_reshaped before prediction: {input_arr_reshaped.shape}")

        # Make predictions
        prediction = model.predict(input_arr_reshaped)

        # Debug print to check the shape of prediction before inverse transformation
        logging.info(f"Shape of prediction before inverse transformation: {prediction.shape}")

        # Inverse transform the predictions to get them back to the original scale
        inverse_prediction = np.exp(prediction)  # Assuming you want to undo np.log scaling

        # Add the inverse transformed predictions to the DataFrame
        df['prediction'] = inverse_prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header=True)
        return prediction_file_path
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise ShippingException(e, sys)
