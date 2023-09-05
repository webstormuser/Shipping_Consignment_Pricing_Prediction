from flask import Flask, render_template,request,redirect,url_for
from flask_cors import CORS, cross_origin
import os ,sys 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import RobustScaler
from shipping_prediction.predictor import ModelResolver
from shipping_prediction.pipeline.predict_pipeline import CustomData,PredictPipeline
from shipping_prediction.utils import load_object
from shipping_prediction.exception import ShippingException 
from shipping_prediction.logger import logging
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route('/')
def home():
    prediction = request.args.get('prediction')  # Get the prediction parameter
    return render_template('home.html', prediction=prediction)

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/prediction/input')
def input_prediction():
    return render_template('input.html')

@app.route('/prediction/batch')
def batch_prediction():
    return render_template('batch_prediction.html')

@app.route('/analysis.html')
def analysis():
    return render_template('analysis.html')

@app.route('/predict_data',methods=['GET', 'POST'])
def predict_data():
    try:
        # get data from request 
        # Get form data from the request
        PO_SO = request.form.get('PO_SO')
        ASN_DN = request.form.get('ASN_DN')
        Sub_Classification = request.form.get('Sub_Classification')
        Line_Item_Quantity = int(request.form.get('Line_Item_Quantity'))
        Pack_Price = float(request.form.get('Pack_Price'))
        Unit_Price = float(request.form.get('Unit_Price'))
        Weight_Kilograms = float(request.form.get('Weight_Kilograms'))
        Freight_Cost_USD = float(request.form.get('Freight_Cost_USD'))
        # Create a CustomData instance with the form data
        custom_data = CustomData(
            PO_SO=PO_SO,
            ASN_DN=ASN_DN,
            Sub_Classification=Sub_Classification,
            Line_Item_Quantity=Line_Item_Quantity,
            Pack_Price=Pack_Price,
            Unit_Price=Unit_Price,
            Weight_Kilograms=Weight_Kilograms,
            Freight_Cost_USD=Freight_Cost_USD
        )

        # Get the data as a DataFrame
        data_frame = custom_data.get_data_as_data_frame()

        # Initialize the PredictPipeline
        predict_pipeline = PredictPipeline()

        # Perform prediction
        prediction = predict_pipeline.predict(data_frame)

        # Redirect back to the home page with the prediction parameter
        return redirect(url_for('home', prediction=prediction[0]))# You can replace this with your actual response
    except Exception as e:
        logging.error(f"Error occured :{str(e)}")
        raise ShippingException(e)

if __name__ == '__main__':
    app.run(port=5002,debug=True)
