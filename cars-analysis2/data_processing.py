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
    df = df.drop_duplicates()

    text_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    drop_cols = ['vin', 'lot', 'Unnamed: 0', 'seats', 'max_power']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

   
    if 'selling_price' in df.columns:
        df = df[df['selling_price'] > 0]
        df = df[df['selling_price'] < 10000000]
    
    if 'km_driven' in df.columns:
        df = df[df['km_driven'] <= 500000]
    
    if 'year' in df.columns:
        df = df[df['year'] >= 1980]
        df = df[df['year'] <= 2026]

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
    
    
    if 'selling_price' in numeric_cols:
        numeric_cols = [col for col in numeric_cols if col != 'selling_price']
    
    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
  
    file_path = Path(__file__).resolve().parent / "data" / "Cars.csv"
    
    df = load_data(file_path)
    print("INSPECTING RAW DATA")
    inspect_data(df)
    
    df_clean = clean_data(df)
    save_data(df_clean, "data/clean_no_encoding.csv")
    print("\nSaved cleaned data (without encoding) to 'data/clean_no_encoding.csv'")
 
    df_encoded = encode_data(df_clean)
  
    df_normalized = normalize_data(df_encoded)
    
   
    save_data(df_normalized, "data/clean.csv")
    print("\nSaved fully processed data (encoded + normalized) to 'data/clean.csv'")
    print(" DATA CLEANING COMPLETED SUCCESSFULLY!")
    print(f"\nFinal dataset shape: {df_normalized.shape}")
    print(f"Features after encoding: {df_normalized.shape[1]}")