# Shipping_Consignment_Pricing_Prediction

# Problem Statement:


The market for logistics analytics is expected to develop at a CAGR of 17.3 percent
from 2019 to 2024, more than doubling in size. This data demonstrates how logistics
organizations are understanding the advantages of being able to predict what will
happen in the future with a decent degree of certainty. Logistics leaders may use this
data to address supply chain difficulties, cut costs, and enhance service levels all at the
same time.

The main goal is to predict the consignment pricing based on the available factors in the
dataset.

# Approach:
The classical machine learning tasks like Data Exploration, Data Cleaning,
Feature Engineering, Model Building and Model Testing. Try out different machine
learning algorithms that’s best fit for the above case.
For this approach I have used XGBRegressor as it is best suited .

# Steps 

* create new environment 
    conda create -p venv python==3.9 

* run reuquirements.txt file (pip install -r requirements.txt)          


# Approach for the project

 1. Data Ingestion :

   * Data is first read as csv and loaded as DataFrame from database .
   * Data is then splited into train and test csv  for validation ,transformation and model training 

2.  Data Validation :

   * Here data is being validated about their distribution either they refer to same distribution in train and test csv refer to base csv 

3. Data Transformation :
   
    * In this phase a ColumnTransformer Pipeline is created.

    *  for Numeric Variables first SimpleImputer is applied with strategy median , then Robust  Scaling is performed on numeric data.
    
    * for Categorical Variables SimpleImputer is applied with most frequent strategy, then onehot encoding  performed , after this data is scaled with robust Scaler.
    
    * This preprocessor is saved as pickle file.


4.  Model Training :

    In this phase base model is tested . The best model found was XGBRegressor .
    This model is saved as pickle file.


5. Prediction Pipeline :

    This pipeline converts given data into DataFrame and has various functions to load pickle files and predict the final results in python.

6. Flask App creation :

    Flask app is created with User Interface to predict consignment shipping price inside a Web Application.

# Docker Setup In EC2 commands to be Executed
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

# Deployemnt Link 
 [url](http://52.23.245.72:8080/)

# Project Demo 

[watch the video here](https://drive.google.com/drive/folders/1uhA13zuZ9ghmrbCAkCE8b_CeqhTC0SPV)
# ScreenShot of Application 

![Image Alt Text](Screenshots/Screenshot%202023-09-08%20at%2011-29-38%20Shipping%20Pricing%20Predictor.png)

![Image Alt Text](Screenshots/Screenshot%202023-09-08%20at%2011-31-16%20Prediction.png)

![Image Alt Text](Screenshots/Screenshot%202023-09-08%20at%2011-31-59%20Input%20Form.png)

![Image Alt Text](Screenshots/Screenshot%202023-09-08%20at%2011-32-33%20Batch%20Prediction.png)

![Image Alt Text](Screenshots/Screenshot%202023-09-08%20at%2011-33-53%20Batch%20Prediction%20-%20Copy.png)


