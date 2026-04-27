def group_by_brand(df):
    return df.groupby('Company Names')['Cars Prices'].mean()

def group_by_fuel(df):
    return df.groupby('Fuel Types')['Cars Prices'].mean()

def group_by_seats(df):
    return df.groupby('Seats')['Cars Prices'].mean()

def group_by_hp(df):
    return df.groupby('HorsePower')['Cars Prices'].mean()

def group_by_engines(df):
    return df.groupby('Engines')['Cars Prices'].mean()