from google.cloud import storage
import pandas as pd
from io import StringIO
from pandas_gbq import to_gbq
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

class ETL:
    def __init__(self, bucket_name, blob_name, project_id, dataset_id, table_id):
        self.bucket_name = bucket_name
        self.blob_name = blob_name
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id

    def extract(self):
        # Set up a client
        client = storage.Client()

        # Get the bucket and blob
        bucket = client.get_bucket(self.bucket_name)
        blob = bucket.get_blob(self.blob_name)

        # Download the data as a string
        data = blob.download_as_string()

        # Load the data into a Pandas DataFrame
        df = pd.read_csv(StringIO(data.decode('utf-8')))

        return df

    def transform(self, df):
        # One-hot encode categorical columns
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
        df = pd.get_dummies(df, columns=categorical_cols)

        # Convert one-hot encoded columns to string dtype
        one_hot_cols = [col for col in df.columns if col.startswith(tuple(categorical_cols))]
        df[one_hot_cols] = df[one_hot_cols].astype(str)

        # Scale continuous columns
        continuous_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
        scaler = MinMaxScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

        return df

    def load(self, df):
        # Load the data into BigQuery
        destination_table = f'{self.project_id}.{self.dataset_id}.{self.table_id}'
        to_gbq(df, destination_table, if_exists='replace')


etl = ETL('datalake-heart-attack', 'heart_dataset.csv', 'predictive-392105', 'heart_attack_processed', 'details')
df = etl.extract()
df = etl.transform(df)
etl.load(df)
