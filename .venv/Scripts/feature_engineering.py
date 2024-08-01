import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath):
    return pd.read_csv(filepath)


def feature_engineering(df):
    # Example: Creating new features
    df['Total SF'] = df['Total Bsmt SF'] + df['Gr Liv Area']

    # Define numerical and categorical features
    num_features = ['Total SF', 'SalePrice']
    cat_features = ['Overall Qual', 'Garage Cars']

    # Check unique values in categorical features
    print("Unique values in 'Overall Qual':", df['Overall Qual'].unique())
    print("Unique values in 'Garage Cars':", df['Garage Cars'].unique())

    # Create transformers for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ],
        remainder='drop'  # Ensure no additional columns are included
    )

    # Fit and transform the data
    X = preprocessor.fit_transform(df)

    # Extract feature names for numerical and categorical features
    num_feature_names = num_features
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = num_feature_names + list(cat_feature_names)

    # Create the transformed DataFrame
    df_transformed = pd.DataFrame(X, columns=feature_names)

    # Print the columns to verify
    print("Transformed columns:", df_transformed.columns)

    # Check for missing columns and add default columns if necessary
    expected_columns = list(df.columns)  # List of all original columns
    missing_columns = set(expected_columns) - set(df_transformed.columns)
    if missing_columns:
        print(f"Warning: Missing columns in the transformed data: {missing_columns}")

    return df_transformed


if __name__ == "__main__":
    df = load_data('../data/cleaned_housing_data.csv')
    df_transformed = feature_engineering(df)
    df_transformed.to_csv('../data/engineered_housing_data.csv', index=False)
    print("Feature Engineering has been successfully completed !!!")
