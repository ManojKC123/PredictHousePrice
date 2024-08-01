import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    # Fill missing values with median or mode
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    return df

def clean_data(df):
    # Example: Removing outliers
    df = df[df['Gr Liv Area'] < 4000]
    return df

if __name__ == "__main__":
    df = load_data('../data/housing_data.csv')
    df = handle_missing_values(df)
    df = clean_data(df)
    df.to_csv('../data/cleaned_housing_data.csv', index=False)
    print("Data Cleaning has been successfully completed !!! ")
