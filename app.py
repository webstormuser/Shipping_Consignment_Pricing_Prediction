from flask import Flask, render_template,request,redirect,url_for,jsonify, send_file
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
import tempfile
from datetime import datetime

load_dotenv()

application= Flask(__name__)
app=application

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

#predict for batch


PREDICTION_DIR = tempfile.mkdtemp()
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        # ... (code to perform predictions and save to a CSV file)
        uploaded_file = request.files['batch_input']
        
        # Check if a file was uploaded
        if 'batch_input' not in request.files or not uploaded_file:
            response = {'error': 'No file uploaded'}
            return jsonify(response), 400

        # Save the uploaded file to a temporary location
        input_file_path = os.path.join(PREDICTION_DIR, uploaded_file.filename)
        uploaded_file.save(input_file_path)
        
        
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
        
        # Construct the URL for downloading the generated CSV file
       # Construct the URL for downloading the generated CSV file
        download_url = url_for('download_prediction', filename=os.path.basename(prediction_file_path))
        # Return the download URL as part of the response
        response = {
            'message': 'Prediction completed successfully.',
            'download_url': download_url,
        }
        return jsonify(response)

    except Exception as e:
        # Handle errors as needed
        error_message = f"Error occurred: {str(e)}"
        response = {'error': error_message}
        return jsonify(response)

@app.route('/download_prediction/<filename>', methods=['GET'])
def download_prediction(filename):
    try:
        prediction_file_path = os.path.join(PREDICTION_DIR, filename)

        if os.path.exists(prediction_file_path):
            return send_file(
                prediction_file_path,
                as_attachment=True,
                download_name=filename,
                mimetype='text/csv'
            )
        else:
            return "File not found", 404
    except Exception as e:
        # Handle errors as needed
        error_message = f"Error occurred: {str(e)}"
        return jsonify({'error': error_message})

    
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0',port=8080)   
    except Exception as e:
        raise ShippingException(e)