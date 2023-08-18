import pymongo
import pandas as pd
import json
from shipping_prediction import utils
# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb+srv://Ashwini:suhan10@cluster0.bkwvjrd.mongodb.net/?retryWrites=true&w=majority")

DATA_FILE_PATH="clean_SCMS_Delivery_History_Dataset.csv"
DATABASE_NAME="clean_Logistic"
COLLECTION_NAME="clean_shipping_data"


if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)


    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    #insert converted json record to mongo db
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)