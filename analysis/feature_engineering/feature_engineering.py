import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape}")
    return df

# Feature 1: Car Age

def add_car_age(df: pd.DataFrame) -> pd.DataFrame:
    current_year = datetime.now().year
    df["car_age"] = current_year - df["year"]
    print(f"[FEATURE] 'car_age' created. Range {df['car_age'].min()} - {df['car_age'].max()} years")
    return df

# Feature 2: Price per Mile 

def add_price_per_mile(df: pd.DataFrame) -> pd.DataFrame:
    df["price_per_mile"] = df["price"] / (df["mileage"] + 1)
    print(f"[FEATURE] 'price_per_mile' created.")
    return df

# Feature 3: Mileage Category (Low / Medium / High)

def add_mileage_category(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 30000, 80000, 150000, float("inf")]
    labels = ["Low", "Medium", "High", "Very High"]
    df["mileage_category"] = pd.cut(df["mileage"], bins=bins, labels=labels)
    print(f"[FEATURE] 'mileage_category' created.")
    print(df["mileage_category"].value_counts().to_string())
    return df

# Feature 4: Age Group (New / Recent / Old / Classic)

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    if "car_age" not in df.columns:
        df = add_car_age(df)

    def classify(age):
        if age <= 3: return "New"
        elif age <= 7: return "Recent"
        elif age <= 15: return "Old"
        else:           return "Classic"

    df["age_group"] = df["car_age"].apply(classify)
    print(f"[FEATURE] 'age_group' created.")
    print(df["age_group"].value_counts().to_string())
    return df

# Feature 5: Brand Tier

def add_brand_tier(df: pd.DataFrame) -> pd.DataFrame:
    luxury_brands = ["audi", "bmw", "cadillac", "infiniti",
                     "jaguar", "lexus", "lincoln", "maserati", "land rover"]
    mid_brands    = ["buick", "chevrolet", "chrysler", "dodge",
                     "ford", "gmc", "honda", "hyundai",
                     "jeep", "kia", "mazda", "nissan",
                     "ram", "toyota"]

    if "brand" in df.columns:
        def assign_tier(brand):
            b = str(brand).lower().strip()
            if any(lb in b for lb in luxury_brands):
                return "Luxury"
            elif any(mb in b for mb in mid_brands):
                return "Mid"
            else:
                return "Economy"

        df["brand_name"] = df["brand"].str.title()
        df["brand_tier"] = df["brand"].apply(assign_tier)

    else:
        brand_cols = [c for c in df.columns if c.startswith("brand_")]
        if not brand_cols:
            df["brand_name"] = "Unknown"
            df["brand_tier"] = "Economy"
            return df

        df["brand_name"] = (
            df[brand_cols].idxmax(axis=1)
            .str.replace("brand_", "", regex=False)
            .str.title()
        )

        luxury_cols = ["brand_audi", "brand_bmw", "brand_cadillac", "brand_infiniti",
                       "brand_jaguar", "brand_lexus", "brand_lincoln", "brand_maserati", "brand_land"]
        mid_cols    = ["brand_buick", "brand_chevrolet", "brand_chrysler", "brand_dodge",
                       "brand_ford", "brand_gmc", "brand_honda", "brand_hyundai",
                       "brand_jeep", "brand_kia", "brand_mazda", "brand_nissan",
                       "brand_ram", "brand_toyota"]

        def assign_tier(row):
            for col in luxury_cols:
                if col in df.columns and row[col] == 1:
                    return "Luxury"
            for col in mid_cols:
                if col in df.columns and row[col] == 1:
                    return "Mid"
            return "Economy"

        df["brand_tier"] = df.apply(assign_tier, axis=1)

    print(f"[FEATURE] 'brand_name' and 'brand_tier' created.")
    print(df["brand_tier"].value_counts().to_string())
    return df

#Feature 6: Salvage Flag

def add_salvage_flag(df: pd.DataFrame) -> pd.DataFrame:
    col = "title_status_salvage insurance"
    if col in df.columns:
        df["is_salvage"] = df[col].astype(int)
        n = df["is_salvage"].sum()
        print(f"[FEATURE] 'is_salvage' created. Salvage cars: {n} ({n/len(df)*100:.1f}%)")
    else:
        df["is_salvage"] = 0
        print("[FEATURE] 'is_salvage' defaulted to 0 — title_status column not found.")
    return df


def run_feature_engineering(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)

    df = add_car_age(df)
    df = add_price_per_mile(df)
    df = add_mileage_category(df)
    df = add_age_group(df)
    df = add_brand_tier(df)
    df = add_salvage_flag(df)

    df.to_csv(output_path, index=False)
    new_cols = ["car_age", "price_per_mile", "mileage_category", "age_group", "brand_name", "brand_tier", "is_salvage"]
    print(f"\n[DONE] Enriched dataset saved to: {output_path}")
    print(f"[INFO] New columns added: {new_cols}")
    return df

if __name__ == "__main__":
    INPUT  = "../data/clean_no_encoding.csv"
    OUTPUT = "../data/featured_cars.csv"
    df = run_feature_engineering(INPUT, OUTPUT)
    print("\nSample output:")
    cols = ["price", "year", "mileage", "car_age", "price_per_mile",
            "mileage_category", "age_group", "brand_name", "brand_tier", "is_salvage"]
    print(df[cols].head(10).to_string())
    


