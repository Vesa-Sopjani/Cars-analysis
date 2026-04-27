import pandas as pd
from pathlib import Path

from stats import basic_statistics
from correlation import correlation_analysis
from grouping import group_by_brand, group_by_fuel, group_by_seats, group_by_engines, group_by_hp

if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent.parent / "data" / "clean_no_encoding.csv"
    df = pd.read_csv(file_path)

    basic_statistics(df)

    corr = correlation_analysis(df)
    print(corr)

    print(group_by_brand(df).sort_values(ascending=False).head(10))    
    print(group_by_fuel(df).sort_values(ascending=False).head(10))  
    print(group_by_seats(df).sort_values(ascending=False).head(10))  
    print(group_by_hp(df).sort_values(ascending=False).head(10))  
    print(group_by_engines(df).sort_values(ascending=False).head(10))  