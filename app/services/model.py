from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


class Model:
    def __init__(self, project_id, dataset_id, table_id):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id

    def get_data(self):
        # Set up a client
        client = bigquery.Client()

        # Query the data from BigQuery
        query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
        """
        df = client.query(query).to_dataframe()

        return df

    def preprocess(self, df):
        # Convert the target variable to numeric data type
        df['output'] = pd.to_numeric(df['output'])

        # Split the data into features and target
        X = df.drop('output', axis=1)
        y = df['output']

        return X, y

    def train(self, X, y):
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train a random forest regressor
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate the mean squared error of the predictions
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean squared error: {mse:.2f}')


model = Model('predictive-392105', 'heart_attack_processed', 'details')
df = model.get_data()
X, y = model.preprocess(df)
model.train(X, y)
