from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from shipping_prediction.predictor import ModelResolver 
import pandas as pd 
from shipping_prediction.utils import load_object 
from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from shipping_prediction.predictor import ModelResolver 
import pandas as pd 
from shipping_prediction.utils import load_object 
import os, sys
from datetime import datetime
import numpy as np

class PredictPipeline:
    """
    This class predicts the shipping price 
    """
    def __init__(self):
        pass
    
    def predict(self):
        try:
            transformer_path = 'saved_models/0/transformer/transformer.pkl'
            model_path = 'saved_models/0/model/model.pkl'
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
        
            # Create an instance of the CustomData class with the new data
            new_data = CustomData(
                PO_SO="SCMS",
                ASN_DN="ASN",
                Sub_Classification="HIV test",
                Line_Item_Quantity=597,
                Pack_Price=32.0,
                Unit_Price=1.6,       
                Weight_Kilograms=209.0,
                Freight_Cost_USD=3088.4,                
            )

            # Convert the new data into a DataFrame
            new_data_df = new_data.get_data_as_data_frame()

            # Transform the new data using the loaded transformer
            new_data_scaled = transformer.transform(new_data_df)

            # Make predictions using the model
            predictions = model.predict(new_data_scaled)

            # Inverse transform the predictions using the target scaler
            original_predictions = np.exp(predictions)-1

            # Print the predicted shipping price
            print("Predicted Shipping Price:", original_predictions[0])

        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            raise ShippingException(str(e), sys.exc_info())

class CustomData:
    def __init__(self,
                PO_SO: str,
                ASN_DN: str,
                Sub_Classification: str,
                Line_Item_Quantity: int,
                Pack_Price: float,
                Unit_Price: float,
                Weight_Kilograms: float,
                Freight_Cost_USD:float):
        self.PO_SO = PO_SO
        self.ASN_DN = ASN_DN
        self.Sub_Classification = Sub_Classification
        self.Line_Item_Quantity = Line_Item_Quantity
        self.Pack_Price = Pack_Price 
        self.Unit_Price = Unit_Price 
        self.Weight_Kilograms = Weight_Kilograms
        self.Freight_Cost_USD =Freight_Cost_USD
        # ...
    def get_data_as_data_frame(self):
        try:
            custom_data_as_input_dict ={
                "PO_SO":[self.PO_SO],
                "ASN_DN":[self.ASN_DN],
                "Sub_Classification":[self.Sub_Classification],
                "Line_Item_Quantity":[self.Line_Item_Quantity],
                "Pack_Price":[self.Pack_Price], 
                "Unit_Price":[self.Unit_Price], 
                "Weight_Kilograms":[self.Weight_Kilograms],
                "Freight_Cost_USD":[self.Freight_Cost_USD],
            }
            return pd.DataFrame(custom_data_as_input_dict)
        except Exception as e:
            logging.error(f" Error occurred : {str(e)}")
            raise ShippingException(e,sys) 

    # ...

# Create an instance of PredictPipeline and call the predict method
pipeline = PredictPipeline()
pipeline.predict()