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
            target_scaler_path = 'saved_models/0/TargetScaler/TargetScaler.pkl'
            target_scaler = load_object(file_path=target_scaler_path)

            # Create an instance of the CustomData class with the new data
            new_data = CustomData(
                PO_SO="SCMS",
                ASN_DN="ASN",
                Country="Haiti",
                Fulfill_Via="Direct Drop",
                Vendor_INCO_Term="EXW",
                Shipment_Mode="Air",
                Sub_Classification="HIV test",
                Unit_of_Measure_Per_Pack=100,
                Line_Item_Quantity=750,
                Pack_Price=71.99,
                Unit_Price=0.72,
                First_Line_Designation="Yes",
                Weight_Kilograms=171,
                Freight_Cost_USD=3518.18,
                Line_Item_Insurance_USD = 86.39,
                Days_to_Process=-502
            )

            # Convert the new data into a DataFrame
            new_data_df = new_data.get_data_as_data_frame()

            # Transform the new data using the loaded transformer
            new_data_scaled = transformer.transform(new_data_df)

            # Make predictions using the model
            predictions = model.predict(new_data_scaled)

            # Inverse transform the predictions using the target scaler
            original_predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))

            # Print the predicted shipping price
            print("Predicted Shipping Price:", original_predictions[0][0])

        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            raise ShippingException(str(e), sys.exc_info())

class CustomData:
    def __init__(self,
                PO_SO: str,
                ASN_DN: str,
                Country: str,
                Fulfill_Via: str,
                Vendor_INCO_Term: str,
                Shipment_Mode: str,
                Sub_Classification: str,
                First_Line_Designation: str,
                Unit_of_Measure_Per_Pack: int,
                Line_Item_Quantity: int,
                Pack_Price: float,
                Unit_Price: float,
                Weight_Kilograms: float,
                Freight_Cost_USD:float,
                Line_Item_Insurance_USD: float,
                Days_to_Process: int):
        self.PO_SO = PO_SO
        self.ASN_DN = ASN_DN
        self.Country = Country
        self.Fulfill_Via = Fulfill_Via
        self.Vendor_INCO_Term = Vendor_INCO_Term
        self.Shipment_Mode = Shipment_Mode 
        self.Sub_Classification = Sub_Classification
        self.First_Line_Designation = First_Line_Designation
        self.Unit_of_Measure_Per_Pack = Unit_of_Measure_Per_Pack
        self.Line_Item_Quantity = Line_Item_Quantity
        self.Pack_Price = Pack_Price 
        self.Unit_Price = Unit_Price 
        self.Weight_Kilograms = Weight_Kilograms
        self.Freight_Cost_USD =Freight_Cost_USD
        self.Line_Item_Insurance_USD = Line_Item_Insurance_USD
        self.Days_to_Process = Days_to_Process
        # ...
    def get_data_as_data_frame(self):
        try:
            custom_data_as_input_dict ={
                "PO_SO":[self.PO_SO],
                "ASN_DN":[self.ASN_DN],
                "Country":[self.Country],
                "Fulfill_Via":[self.Fulfill_Via],
                "Vendor_INCO_Term":[self.Vendor_INCO_Term],
                "Shipment_Mode":[self.Shipment_Mode], 
                "Sub_Classification":[self.Sub_Classification],
                "First_Line_Designation":[self.First_Line_Designation],
                "Unit_of_Measure_Per_Pack":[self.Unit_of_Measure_Per_Pack],
                "Line_Item_Quantity":[self.Line_Item_Quantity],
                "Pack_Price":[self.Pack_Price], 
                "Unit_Price":[self.Unit_Price], 
                "Weight_Kilograms":[self.Weight_Kilograms],
                "Freight_Cost_USD":[self.Freight_Cost_USD],
                "Line_Item_Insurance_USD":[self.Line_Item_Insurance_USD],
                "Days_to_Process":[self.Days_to_Process]
            }
            return pd.DataFrame(custom_data_as_input_dict)
        except Exception as e:
            logging.error(f" Error occurred : {str(e)}")
            raise ShippingException(e,sys) 

    # ...

# Create an instance of PredictPipeline and call the predict method
pipeline = PredictPipeline()
pipeline.predict()
