import pandas as pd

def group_by_brand(df):
    return df.groupby('brand')['price'].mean()

def group_by_model(df):
    return df.groupby('model')['price'].mean()

def group_by_year(df):
    return df.groupby('year')['price'].mean()

def group_by_mileage(df):
    return df.groupby('mileage')['price'].mean()

def group_by_condition(df):
    return df.groupby('condition')['price'].mean()

def group_by_title(df):
    return df.groupby('title_status')['price'].mean()

def group_by_age(df):
    return df.groupby('age')['price'].mean()

def mileage_bins(df):
    df['mileage_group'] = pd.cut(df['mileage'], bins=5)
    return df.groupby('mileage_group')['price'].mean()

def top_expensive_brands(df):
    return df.groupby('brand')['price'].mean().sort_values(ascending=False).head(5)

def cheapest_brands(df):
    return df.groupby('brand')['price'].mean().sort_values().head(5)