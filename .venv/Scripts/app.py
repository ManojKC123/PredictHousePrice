import pandas as pd
from joblib import load


def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)


def add_total_sf(df):
    """Add the Total SF feature to the DataFrame."""
    df['Total SF'] = df['1st Flr SF'] + df['2nd Flr SF'] + df['Total Bsmt SF']
    return df


def align_columns(df, transformer_columns):
    """Ensure the DataFrame has all columns required by the transformer."""
    missing_cols = set(transformer_columns) - set(df.columns)

    # Add missing columns with default values
    for col in missing_cols:
        if 'cat__' in col:
            df[col] = 'Missing'
        else:
            df[col] = 0

    # Reorder columns to match the transformer
    df = df[transformer_columns]
    return df


def predict(model, df):
    """Predict the target using the trained model."""
    # Add Total SF column
    df = add_total_sf(df)

    # Drop the SalePrice column if it exists
    if 'SalePrice' in df.columns:
        df = df.drop('SalePrice', axis=1)

    # Get the expected columns from the model
    transformer_columns = model.named_steps['preprocessor'].get_feature_names_out()

    # Align columns with model expectations
    df = align_columns(df, transformer_columns)

    # Make predictions
    predictions = model.predict(df)
    return predictions


if __name__ == "__main__":
    # Load the model
    model = load('../models/housing_model.joblib')  # Adjust the path as necessary

    # Load the input data
    df = load_data('../data/new_data.csv')  # Adjust the path as necessary

    # Predict and print the results
    predictions = predict(model, df)
    print(predictions)
