import pandas as pd
from pathlib import Path

def load_data(path):
    df = pd.read_csv(path, encoding='latin1')
    return df


def inspect_data(df):
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())
    print("\nData Types:\n", df.dtypes)



def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize text columns
    text_cols = ['brand', 'model', 'color', 'state', 'country', 'title_status', 'condition']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Remove unnecessary columns
    drop_cols = ['vin', 'lot', 'Unnamed: 0']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Remove unrealistic values (outliers)
    df = df[df['price'] < 100000]
    df = df[df['mileage'] < 500000]
    df = df[df['year'] > 1980]

    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("unknown")

    return df



def encode_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df



def normalize_data(df):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def save_data(df, path):
    df.to_csv(path, index=False)



if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent /  "data" / "Cars.csv"

    # Load
    df = load_data(file_path)

    # Inspect
    inspect_data(df)

    # Clean
    df = clean_data(df)
    save_data(df, "data/clean_no_encoding.csv")

    # Encode + Normalize
    df = encode_data(df)
    df = normalize_data(df)

    # Save final
    save_data(df, "data/clean.csv")

    print("\n✅ Data cleaning completed successfully!")