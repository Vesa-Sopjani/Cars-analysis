def group_by_brand(df):
    return df.groupby('brand')['price'].mean()

def group_by_model(df):
    return df.groupby('model')['price'].mean()

def group_by_year(df):
    return df.groupby('year')['price'].mean()

def group_by_mileage(df):
    return df.groupby('mileage')['price'].mean()
