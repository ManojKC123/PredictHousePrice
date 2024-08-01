import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def train_model(df):
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # Identify categorical columns
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

    # Identify numerical columns
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

    # Preprocessing for numerical data
    numerical_transformer = 'passthrough'

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the model with the best parameters
    model = RandomForestRegressor(
        random_state=42,
        max_depth=20,
        max_features='sqrt',
        n_estimators=300
    )

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f'Mean Absolute Error: {mae}')

    # Ensure the models directory exists
    os.makedirs('../models', exist_ok=True)

    # Save the model to a file
    dump(pipeline, '../models/housing_model.joblib')

    return pipeline

if __name__ == "__main__":
    df = load_data('../data/engineered_housing_data.csv')
    model = train_model(df)
