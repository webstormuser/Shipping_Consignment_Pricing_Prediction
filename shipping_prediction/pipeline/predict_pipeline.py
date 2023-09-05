from shipping_prediction.exception import ShippingException
from shipping_prediction.logger import logging
from shipping_prediction.predictor import ModelResolver
import pandas as pd
from shipping_prediction.utils import load_object
import os,sys
from datetime import datetime

class PredictPipeline:
    '''
     This class predict either the patient has Thyroid or not 
    '''
    def __init__(self):pass

    def predict(self,features):
        try:
            transformer_path = 'saved_models/0/transformer/transformer.pkl'
            model_path = 'saved_models/0/model/model.pkl'
            transformer = load_object(file_path = transformer_path)
            model = load_object(file_path = model_path)
            prediction = model.predict(data_scaled)
            original_prediction =np.exp(prediction)-1
            return original_prediction          
        except Exception as e :
            raise ThyroidException(e,sys)
        
        
class CustomData:
    def __init__(self, 
                      PO_SO:str,
                      ASN_DN:str,
                      Sub_Classification:str,
                      Line_Item_Quantity:int,
                      Pack_Price:float,
                      Unit_Price:float,
                      Weight_Kilograms:float,
                      Freight_Cost_USD:float
                      ):
                    self.PO_SO = PO_SO
                    self.ASN_DN = ASN_DN
                    self.Sub_Classification=Sub_Classification
                    self.Line_Item_Quantity=Line_Item_Quantity
                    self.Pack_Price=Pack_Price
                    self.Unit_Price=Unit_Price
                    self.Weight_Kilograms=Weight_Kilograms
                    self.Freight_Cost_USD                
                      
    def get_data_as_data_frame(self):
        try:
            custom_data_as_input_dict={
                "PO_SO":[self.PO_SO],
                "ASN_DN":[self.ASN_DN],
                "Sub_Classification":[self.Sub_Classification],
                "Line_Item_Quantity":[self.Line_Item_Quantity],
                "Pack_Price":[self.Pack_Price],
                "Unit_Price":[self.Unit_Price],
                "Weight_Kilograms":[self.Weight_kilograms],
                "Freight_Cost_USD":[self.Freight_Cost_USD]                
            }
            return pd.DataFrame(custom_data_as_input_dict)

        except Exception as e:
                raise ShippingException(e,sys)