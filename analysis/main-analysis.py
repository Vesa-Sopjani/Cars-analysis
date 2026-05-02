import pandas as pd
from pathlib import Path

from stats import basic_statistics
from correlation import correlation_analysis
from priceDistribution import price_distribution
from grouping import group_by_brand, group_by_model, group_by_year, group_by_mileage, mileage_bins, top_expensive_brands, cheapest_brands, group_by_car_age, group_by_age_group, group_by_mileage_category, group_by_brand_tier, group_by_salvage

if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent.parent / "data" / "featured_cars.csv"
    df = pd.read_csv(file_path)

    basic_statistics(df)

    price_distribution(df)

    corr = correlation_analysis(df)

    print(group_by_brand(df).sort_values(ascending=False).head(10))    
    print(group_by_model(df).sort_values(ascending=False).head(10))  
    print(group_by_year(df).sort_values(ascending=False).head(10))  
    print(group_by_car_age(df).sort_values(ascending=False).head(10))  
    print(group_by_mileage(df).sort_values(ascending=False).head(10))  
    print(group_by_age_group(df).sort_values(ascending=False).head(10))  
    print(group_by_mileage_category(df).sort_values(ascending=False).head(10))  
    print(group_by_brand_tier(df).sort_values(ascending=False).head(10))  
    print(group_by_salvage(df).sort_values(ascending=False).head(10))  
    print(mileage_bins(df))
    print(top_expensive_brands(df))
    print(cheapest_brands(df))