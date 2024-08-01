import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(data, numerical_features, categorical_features):
    """Preprocess data by splitting features and target, and setting up preprocessing pipelines."""
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
    ])

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor


def build_model(preprocessor):
    """Create and return a machine learning pipeline with preprocessing and model."""
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])
    return model


def train_model(model, X_train, y_train):
    """Train the model using the training data."""
    model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using the test data and return the mean squared error."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)


def load_model(filename):
    """Load a model from a file."""
    return joblib.load(filename)


def make_predictions(model, new_data):
    """Make predictions using the loaded model and new data."""
    return model.predict(new_data)


def main():
    # File paths
    training_file_path = '../data/housing_data.csv'  # Replace with your training data CSV path
    new_data_file_path = '../data/new_data.csv'  # Replace with your new data CSV path
    model_file_path = 'housing_model.pkl'

    # Define features
    numerical_features = [
        'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built',
        'Year Remod/Add', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
        'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF',
        'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath',
        'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr',
        'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch',
        'Screen Porch', 'Pool Area', 'Misc Val', 'Garage Cars', 'Garage Area'
    ]

    categorical_features = [
        'MS SubClass', 'MS Zoning', 'Street', 'Alley', 'Lot Shape',
        'Land Contour', 'Utilities', 'Lot Config', 'Land Slope', 'Neighborhood',
        'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Roof Style',
        'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Exter Qual', 'Exter Cond',
        'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1',
        'BsmtFin Type 2', 'Heating', 'Heating QC', 'Central Air', 'Electrical',
        'Kitchen Qual', 'Functional', 'Fireplace Qu', 'Garage Type', 'Garage Finish',
        'Garage Qual', 'Garage Cond', 'Paved Drive', 'Fence', 'Misc Feature',
        'Sale Type', 'Sale Condition'
    ]

    # Load and preprocess training data
    training_data = load_data(training_file_path)
    X, y, preprocessor = preprocess_data(training_data, numerical_features, categorical_features)

    # Build and train the model
    model = build_model(preprocessor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_model(model, X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, model_file_path)
    print(f'Model saved to {model_file_path}')

    # Load the model
    loaded_model = load_model(model_file_path)

    # Load and prepare new data
    new_data = load_data(new_data_file_path)
    new_data = new_data[X.columns]  # Ensure the new data has the same columns as the training data

    # Make predictions
    predictions = make_predictions(loaded_model, new_data)
    print('Predictions for new data:', predictions)


if __name__ == "__main__":
    main()
