import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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
    print("DataFrame Columns:", df.columns.tolist())
    print("DataFrame Shape:", df.shape)

    # Define target and features
    y = df['SalePrice']
    X = df.drop('SalePrice', axis=1)

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

    # Define the model
    model = RandomForestRegressor(random_state=42)

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Define hyperparameters for GridSearchCV
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Predict and calculate MAE
    predictions = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')

    # Ensure the models directory exists
    os.makedirs('../models', exist_ok=True)

    # Save the model to a file
    dump(best_model, '../models/housing_model.joblib')

    return best_model


if __name__ == "__main__":
    df = load_data('../data/engineered_housing_data.csv')
    model = train_model(df)
