import pandas as pd
from pathlib import Path


def load_data(path):
    df = pd.read_csv(path)
    return df

def inspect_data(df):
    print("shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:\n", df.duplicated().sum())

def convert_to_numeric(df):
    def fix_seats(value):
        try:
            if '+' in str(value):
                parts = value.split('+')
                return sum(int(p) for p in parts)
            return int(value)
        except:
            return None
        
    df['Seats']= df['Seats'].apply(fix_seats)

    numeric_columns= [
        'CC/Battery Capacity',
        'HorsePower',
        'Total Speed',
        'Performance(0 - 100 )KM/H',
        'Cars Prices',
        'Torque'
    ]
    for col in numeric_columns:
        df[col]= df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)

        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def clean_data(df):
    df=df.drop_duplicates()

    df['Company Names'] = df['Company Names'].str.upper().str.strip()
    df['Fuel Types'] = df['Fuel Types'].str.lower().str.strip()

    df = df[df['Cars Prices'] < 1e7]
    df = df[df['HorsePower'] < 2000]
    df = df[df['Seats'] <= 10]

    df['Fuel Types'] = df['Fuel Types'].replace({
        'plug in hyrbrid': 'plug-in hybrid',
        'hybrid (petrol)': 'hybrid',
        'hybrid/electric': 'hybrid',
        'petrol/hybrid': 'hybrid',
    })

    df = df[df['CC/Battery Capacity'] < 10000]
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col]= df[col].fillna(df[col].mean())
        else:
            df[col]= df[col].fillna("Unknown")

    return df

def encode_data(df):
    df= pd.get_dummies(df, drop_first=True)
    return df

def normalize_data(df):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    numeric_cols=df.select_dtypes(include=['int64', 'float64']).columns

    df[numeric_cols]=scaler.fit_transform(df[numeric_cols])
    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent.parent / "Cars-analysis" / "data" / "Cars.csv"
    df = pd.read_csv(file_path,  encoding='latin1', delimiter=',')
    df = convert_to_numeric(df) 
    inspect_data(df)

    df=clean_data(df)
    save_data(df, "data/clean_no_encoding.csv")

    df=encode_data(df)
    df=normalize_data(df)

    save_data(df, "data/clean.csv")

    print("data cleaning completed!")