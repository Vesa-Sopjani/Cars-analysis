import pandas as pd

def group_by_brand(df):
    return df.groupby('brand_name')['price'].mean()

def group_by_model(df):
    return df.groupby('model')['price'].mean()

def group_by_year(df):
    return df.groupby('year')['price'].mean()

def group_by_car_age(df):
    return df.groupby('car_age')['price'].mean()

def group_by_age_group(df):
    return df.groupby('age_group')['price'].mean()

def group_by_mileage(df):
    return df.groupby('mileage')['price'].mean()

def group_by_mileage_category(df):
    return df.groupby('mileage_category')['price'].mean()

def group_by_brand_tier(df):
    return df.groupby('brand_tier')['price'].mean()

def group_by_salvage(df):
    return df.groupby('is_salvage')['price'].mean()

def mileage_bins(df):
    df['mileage_group'] = pd.cut(df['mileage'], bins=5)
    return df.groupby('mileage_group')['price'].mean()

def top_expensive_brands(df):
    return df.groupby('brand_name')['price'].mean().sort_values(ascending=False).head(5)

def cheapest_brands(df):
    return df.groupby('brand_name')['price'].mean().sort_values().head(5)