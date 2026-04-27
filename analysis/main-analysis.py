import pandas as pd
from pathlib import Path

from stats import basic_statistics
from correlation import correlation_analysis
from grouping import group_by_brand, group_by_model, group_by_year, group_by_mileage

if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent.parent / "data" / "clean_no_encoding.csv"
    df = pd.read_csv(file_path)

    basic_statistics(df)

    corr = correlation_analysis(df)
    print(corr)

    print(group_by_brand(df).sort_values(ascending=False).head(10))    
    print(group_by_model(df).sort_values(ascending=False).head(10))  
    print(group_by_year(df).sort_values(ascending=False).head(10))  
    print(group_by_mileage(df).sort_values(ascending=False).head(10))  